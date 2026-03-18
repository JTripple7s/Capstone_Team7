#!/usr/bin/env python3
"""
Ball Tracker with Chassis Following - HiWonder TurboPi
Supports: Red, Green, Blue balls
Features: Camera tracking + Chassis following with mecanum wheels
"""
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
import sdk.pid as pid
import sdk.yaml_handle as yaml_handle

class BallTracker(Node):
    def __init__(self):
        super().__init__('ball_tracker')

        # Publishers
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, '/ball_debug', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber
        self.create_subscription(Image, '/image_raw', self.image_callback, 1)

        # Load LAB color data from HiWonder config
        self.lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)

        # Target colors - scan for ANY of these colors
        self.target_colors = ['red', 'green', 'blue']
        self.current_color = 'green'  # Default, will change when ball is detected

        # Processing resolution (HiWonder standard)
        self.pro_size = (320, 240)

        # === CAMERA PID (HiWonder exact) ===
        self.servo_x_pid = pid.PID(P=0.25, I=0.05, D=0.009)
        self.servo_y_pid = pid.PID(P=0.25, I=0.05, D=0.009)

        # === CHASSIS PID (tuned for aggressive alignment) ===
        # Higher P = faster rotation to face ball
        self.chassis_yaw_pid = pid.PID(P=0.015, I=0.005, D=0.001)   # Rotation (increased for faster alignment)
        self.chassis_dist_pid = pid.PID(P=0.006, I=0.003, D=0.0001) # Distance

        # Servo state
        self.servo_x = 1500
        self.servo_y = 2100  # Start looking straight ahead (not ceiling)

        # HiWonder servo limits
        self.servo_min_x, self.servo_max_x = 800, 2200
        self.servo_min_y, self.servo_max_y = 1200, 2400

        # Dead zones (HiWonder exact)
        self.pan_tilt_x_threshold = 15
        self.pan_tilt_y_threshold = 15

        # Tracking state
        self.last_color_circle = None
        self.lost_target_count = 0

        # Ball size filter (lower values = detect farther balls)
        self.min_radius = 15  # Reduced to detect balls when far away
        self.min_area = 300   # Reduced for far detection

        # Search behavior
        self.scan_direction = 1
        self.search_speed = 8

        # === CHASSIS FOLLOWING ===
        self.chassis_enabled = True
        self.target_radius = 65  # Desired ball size (pixels) - larger = closer following
        self.radius_deadzone = 8  # +/- pixels tolerance for distance
        self.servo_center = 1500  # Servo center position
        self.align_threshold = 40  # Servo error threshold for "aligned"
        self.approach_gain = 0.012  # Speed gain for approach (higher = faster)
        self.min_approach_speed = 0.15  # Minimum speed when too far

        # === SMOOTHING (HiWonder's smooth_value approach) ===
        self.smoothing_alpha = 0.5  # EMA smoothing factor (HiWonder uses 0.5)
        self.last_x = None
        self.last_y = None
        self.last_angular_z = 0.0
        self.last_linear_x = 0.0

        # Color RGB for drawing
        self.range_rgb = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
        }

        self.publish_servo(self.servo_x, self.servo_y)
        self.get_logger().info(f"Ball tracker ready - tracking ANY color: {self.target_colors} (chassis following: {self.chassis_enabled})")

    def publish_servo(self, servo_x, servo_y):
        servo_x = int(np.clip(servo_x, self.servo_min_x, self.servo_max_x))
        servo_y = int(np.clip(servo_y, self.servo_min_y, self.servo_max_y))

        msg = SetPWMServoState()
        msg.duration = 0.02
        state_x = PWMServoState()
        state_x.id, state_x.position, state_x.offset = [2], [servo_x], [0]
        state_y = PWMServoState()
        state_y.id, state_y.position, state_y.offset = [1], [servo_y], [0]
        msg.state = [state_x, state_y]
        self.servo_pub.publish(msg)
        self.servo_x, self.servo_y = servo_x, servo_y

    def publish_velocity(self, linear_x, linear_y, angular_z):
        """Publish chassis velocity for mecanum wheels"""
        twist = Twist()
        twist.linear.x = float(np.clip(linear_x, -0.5, 0.5))
        twist.linear.y = float(np.clip(linear_y, -0.5, 0.5))
        twist.angular.z = float(np.clip(angular_z, -1.5, 1.5))
        self.cmd_vel_pub.publish(twist)

    def stop_chassis(self):
        """Stop all chassis movement"""
        self.publish_velocity(0.0, 0.0, 0.0)

    def smooth_value(self, current, last, alpha=0.5):
        """HiWonder's smooth_value - EMA smoothing"""
        if last is None:
            return current
        return alpha * current + (1 - alpha) * last

    def smooth_position(self, x, y):
        """Apply EMA smoothing to ball position (reduces jitter)"""
        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return x, y
        smoothed_x = self.smooth_value(x, self.last_x, self.smoothing_alpha)
        smoothed_y = self.smooth_value(y, self.last_y, self.smoothing_alpha)
        self.last_x, self.last_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

    def image_callback(self, msg):
        try:
            # Convert image
            if "yuy" in msg.encoding.lower():
                frame = cv2.cvtColor(np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
            else:
                frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

            h, w = frame.shape[:2]
            result_image = frame.copy()

            # === HiWonder's EXACT processing ===
            image = cv2.resize(frame, self.pro_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            image = cv2.GaussianBlur(image, (5, 5), 5)

            # === MULTI-COLOR DETECTION: Scan for ANY ball color ===
            best_circle = None
            best_area = 0
            detected_color = None

            for color in self.target_colors:
                # Get color range from HiWonder's lab_config.yaml
                min_color = self.lab_data[color]['min']
                max_color = self.lab_data[color]['max']

                # Create mask for this color
                mask = cv2.inRange(image, tuple(min_color), tuple(max_color))

                # HiWonder's morphological ops
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

                # Find contours
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

                # Filter by area AND circularity (to avoid detecting text/rectangles)
                for contour in contours:
                    area = math.fabs(cv2.contourArea(contour))
                    if area < self.min_area:
                        continue

                    # Calculate circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * math.pi * area / (perimeter * perimeter)

                    # Only accept shapes that are at least 50% circular (balls, not text)
                    if circularity < 0.5:
                        continue

                    # Weight by both area and circularity
                    score = area * circularity

                    if score > best_area:
                        (cx, cy), r = cv2.minEnclosingCircle(contour)
                        best_circle = ((cx, cy), r)
                        best_area = score
                        detected_color = color

            circle = best_circle
            if detected_color:
                self.current_color = detected_color

            # Process detection
            if circle is not None:
                (cx, cy), r = circle
                x = cx / self.pro_size[0] * w
                y = cy / self.pro_size[1] * h
                r_scaled = r / self.pro_size[0] * w

                # Filter small objects
                if r_scaled < self.min_radius:
                    circle = None

            # Valid ball - track it
            if circle is not None:
                self.lost_target_count = 0
                r = r_scaled
                self.last_color_circle = circle

                # Store RAW position for display (no lag)
                raw_x, raw_y = x, y

                # === POSITION SMOOTHING for CONTROL only (reduces jitter in PID) ===
                ctrl_x, ctrl_y = self.smooth_position(x, y)

                # === CAMERA TRACKING (PID) - use smoothed position for control ===
                track_x = w/2 if abs(ctrl_x - w/2) < self.pan_tilt_x_threshold else ctrl_x
                track_y = h/2 if abs(ctrl_y - h/2) < self.pan_tilt_y_threshold else ctrl_y

                # X-axis servo (pan)
                self.servo_x_pid.SetPoint = w / 2
                self.servo_x_pid.update(track_x)
                pid_x_output = self.servo_x_pid.output  # Save for chassis control
                self.servo_x += int(pid_x_output)

                # Y-axis servo (tilt)
                self.servo_y_pid.SetPoint = h / 2
                self.servo_y_pid.update(track_y)
                pid_y_output = self.servo_y_pid.output
                self.servo_y -= int(pid_y_output)

                self.publish_servo(self.servo_x, self.servo_y)

                # === CHASSIS FOLLOWING (Tank-like approach + alignment) ===
                if self.chassis_enabled:
                    # --- ROTATION: Align body with ball ---
                    servo_error = self.servo_x - self.servo_center  # Positive = looking right

                    # Use BOTH servo error AND PID output for robust alignment
                    angular_z = (servo_error * 0.004) + (pid_x_output * 0.025)
                    angular_z = np.clip(angular_z, -1.8, 1.8)

                    # --- DISTANCE: Approach and maintain target distance ---
                    radius_error = r - self.target_radius  # Positive = too close, Negative = too far

                    if abs(radius_error) > self.radius_deadzone:
                        # Calculate approach/retreat speed
                        linear_x = -radius_error * self.approach_gain

                        # Ensure minimum speed when ball is far (aggressive approach)
                        if radius_error < -self.radius_deadzone:  # Ball is too far
                            linear_x = max(linear_x, self.min_approach_speed)

                        linear_x = np.clip(linear_x, -0.35, 0.4)  # Allow faster forward
                    else:
                        linear_x = 0.0

                    # Deadzone for rotation ONLY when aligned AND at correct distance
                    if abs(servo_error) < self.align_threshold and abs(pid_x_output) < 5:
                        angular_z = 0.0

                    # === VELOCITY SMOOTHING (less smoothing = more responsive) ===
                    linear_x = self.smooth_value(linear_x, self.last_linear_x, 0.6)
                    angular_z = self.smooth_value(angular_z, self.last_angular_z, 0.6)
                    self.last_linear_x = linear_x
                    self.last_angular_z = angular_z

                    # Check alignment status
                    is_aligned = abs(servo_error) < self.align_threshold

                    self.publish_velocity(linear_x, 0.0, angular_z)

                # Draw circle at RAW position (no lag - accurate to actual ball)
                draw_color = self.range_rgb.get(self.current_color, (0, 255, 0))
                cv2.circle(result_image, (int(raw_x), int(raw_y)), int(r), draw_color, 2)
                cv2.putText(result_image, f"TRACKING {self.current_color.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
                cv2.putText(result_image, f"r={int(r)}", (int(raw_x)-20, int(raw_y)-int(r)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show chassis/alignment status
                if self.chassis_enabled:
                    # Alignment status
                    if is_aligned:
                        align_text = "ALIGNED"
                        align_color = (0, 255, 0)
                    else:
                        direction = "R" if servo_error > 0 else "L"
                        align_text = f"TURN {direction}"
                        align_color = (0, 165, 255)

                    cv2.putText(result_image, align_text, (w-150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, align_color, 2)

                    # Distance status
                    if abs(radius_error) <= self.radius_deadzone:
                        dist_text = "DIST OK"
                        dist_color = (0, 255, 0)
                    elif radius_error > 0:
                        dist_text = "TOO CLOSE"
                        dist_color = (0, 0, 255)
                    else:
                        dist_text = "TOO FAR"
                        dist_color = (255, 165, 0)

                    cv2.putText(result_image, dist_text, (w-150, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 2)

                    cv2.putText(result_image, f"ang={angular_z:.2f} lin={linear_x:.2f}", (w-200, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                # Ball lost
                self.lost_target_count += 1
                if self.lost_target_count > 10:
                    self.last_color_circle = None

                # Stop chassis when lost
                self.stop_chassis()

                # Search behavior
                if self.lost_target_count > 5:
                    self.servo_y = 2400  # Look at floor
                    self.servo_x += self.search_speed * self.scan_direction

                    if self.servo_x >= self.servo_max_x - 50:
                        self.scan_direction = -1
                    elif self.servo_x <= self.servo_min_x + 50:
                        self.scan_direction = 1

                    self.publish_servo(self.servo_x, self.servo_y)

                cv2.putText(result_image, "SEARCHING", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Show servo position
            cv2.putText(result_image, f"pan={self.servo_x} tilt={self.servo_y}",
                       (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show target color
            # Show current tracking status
            if self.current_color:
                cv2.putText(result_image, f"Tracking: {self.current_color}", (10, h-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[self.current_color], 2)
            else:
                cv2.putText(result_image, "Scanning: R/G/B", (10, h-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Publish debug image
            out = Image()
            out.header = msg.header
            out.height, out.width = h, w
            out.encoding = "bgr8"
            out.step = w * 3
            out.data = result_image.tobytes()
            self.debug_pub.publish(out)

        except Exception as e:
            self.get_logger().error(str(e))

def main():
    rclpy.init()
    node = BallTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Stop motors on exit
    try:
        node.stop_chassis()
        node.destroy_node()
    except:
        pass
    try:
        rclpy.shutdown()
    except:
        pass

if __name__ == '__main__':
    main()
