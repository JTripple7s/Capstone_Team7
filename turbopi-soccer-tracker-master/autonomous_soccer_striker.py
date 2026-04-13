#!/usr/bin/env python3
"""
Autonomous Soccer Striker - HiWonder TurboPi
============================================
Identifies a ball and a goal using YOLO, then positions itself 
to score the ball into the goal.

States:
  - SEARCH_BALL: Scans for the ball.
  - APPROACH_BALL: Moves toward the ball.
  - SEARCH_GOAL: Once near the ball, scans for the goal.
  - POSITIONING: Strafes/aligns so Robot-Ball-Goal are in a line.
  - STRIKE: Moves forward rapidly to score.

Author: Gemini CLI (based on Capstone Team 7 work)
"""

import cv2
import math
import numpy as np
import sys
import serial
import struct
import subprocess
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import SetPWMServoState

# Import PID from the project's SDK
# Assuming the path is correct relative to where this script runs
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/driver/sdk/sdk")
try:
    import sdk.pid as pid
except ImportError:
    # Fallback if local import fails
    class MockPID:
        def __init__(self, P=0, I=0, D=0):
            self.SetPoint = 0
            self.output = 0
        def update(self, current): pass
    pid = type('module', (), {'PID': MockPID})

# ============ SERIAL COMMUNICATION (Copied from yolo_soccer_tracker.py) ============
CRC8_TABLE = [
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53
]

SERIAL_PORT = None

def init_serial():
    global SERIAL_PORT
    try:
        SERIAL_PORT = serial.Serial("/dev/rrc", baudrate=1000000, timeout=5)
        SERIAL_PORT.rts = False
        SERIAL_PORT.dtr = False
        return True
    except Exception as e:
        print(f"Serial init failed: {e}")
        return False

def checksum_crc8(data):
    check = 0
    for b in data:
        check = CRC8_TABLE[check ^ b]
    return check & 0x00FF

def buf_write(func, data):
    global SERIAL_PORT
    if SERIAL_PORT is None:
        return
    buf = [0xAA, 0x55, int(func)]
    buf.append(len(data))
    buf.extend(data)
    buf.append(checksum_crc8(bytes(buf[2:])))
    try:
        SERIAL_PORT.write(bytes(buf))
    except:
        pass

def direct_servo_set(duration, positions):
    duration_ms = int(duration * 1000)
    data = [0x01, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, len(positions)]
    for servo_id, pos in positions:
        data.extend(struct.pack("<BH", servo_id, pos))
    buf_write(4, data)

def direct_motor_set(motor_speeds):
    data = [0x05, len(motor_speeds)]
    for motor_id, duty in motor_speeds:
        data.extend(struct.pack("<Bf", motor_id - 1, float(duty)))
    buf_write(3, data)

def mecanum_drive(linear_x, linear_y, angular_z, max_duty=80):
    vx = linear_x * max_duty
    vy = linear_y * max_duty
    wz = angular_z * max_duty
    fl = vx - vy - wz
    fr = vx + vy + wz
    rl = vx + vy - wz
    rr = vx - vy + wz
    m1, m2, m3, m4 = -fl, fr, -rl, rr
    m1 = max(-max_duty, min(max_duty, m1))
    m2 = max(-max_duty, min(max_duty, m2))
    m3 = max(-max_duty, min(max_duty, m3))
    m4 = max(-max_duty, min(max_duty, m4))
    direct_motor_set([[1, m1], [2, m2], [3, m3], [4, m4]])

# ============ BALL DISTANCE ESTIMATION ============
class BallDistanceEstimator:
    def __init__(self, focal_length_pixels=450.0, ball_diameter_cm=6.5):
        self.focal_length = focal_length_pixels
        self.ball_diameter_cm = ball_diameter_cm
        self.last_distance = None
        self.smooth_alpha = 0.3

    def estimate_distance(self, ball_radius_pixels):
        if ball_radius_pixels <= 0: return None
        ball_diameter_pixels = ball_radius_pixels * 2
        distance_cm = (self.ball_diameter_cm * self.focal_length) / ball_diameter_pixels
        if self.last_distance is not None:
            distance_cm = self.smooth_alpha * distance_cm + (1 - self.smooth_alpha) * self.last_distance
        self.last_distance = distance_cm
        return distance_cm

# ============ AUTONOMOUS SOCCER STRIKER NODE ============
class AutonomousSoccerStriker(Node):
    def __init__(self):
        super().__init__('autonomous_soccer_striker')
        
        # Configuration
        self.declare_parameter('model_path', '/home/ubuntu/best.pt')
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        # YOLO Setup
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO: {e}")
            self.model = None

        # Publishers & Subscribers
        self.debug_pub = self.create_publisher(Image, '/soccer/debug_image', 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 1)

        # Servo Control
        self.servo_x = 1500  # Pan
        self.servo_y = 2500  # Tilt (Down)
        self.servo_x_pid = pid.PID(P=0.20, I=0.008, D=0.008)
        self.servo_y_pid = pid.PID(P=0.40, I=0.015, D=0.015)
        
        # Chassis Control
        self.chassis_yaw_pid = pid.PID(P=0.012, I=0.002, D=0.08)
        self.last_linear_x = 0.0
        self.last_linear_y = 0.0
        self.last_angular_z = 0.0

        # State Machine
        self.state = 'SEARCH_BALL'
        self.state_start_time = time.time()
        self.lost_ball_count = 0
        self.lost_goal_count = 0
        self.scan_direction = 1
        
        # Distance Estimator
        self.distance_estimator = BallDistanceEstimator()
        
        # Tracking Targets
        self.ball_target = None
        self.goal_target = None
        
        # Serial Init
        self.use_serial = init_serial()
        if self.use_serial:
            direct_servo_set(1.0, [[1, self.servo_y], [2, self.servo_x]])

    def set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"Transitioning: {self.state} -> {new_state}")
            self.state = new_state
            self.state_start_time = time.time()

    def set_velocity(self, lx, ly, az):
        # Smoothing
        lx = 0.7 * lx + 0.3 * self.last_linear_x
        ly = 0.7 * ly + 0.3 * self.last_linear_y
        az = 0.7 * az + 0.3 * self.last_angular_z
        self.last_linear_x, self.last_linear_y, self.last_angular_z = lx, ly, az
        if self.use_serial:
            mecanum_drive(lx, ly, az)

    def image_callback(self, msg):
        if self.model is None: return
        
        # Convert Image
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        h, w = frame.shape[:2]
        
        # YOLO Detection
        results = self.model(frame, conf=0.4, verbose=False)
        detections = []
        self.ball_target = None
        self.goal_target = None
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = self.model.names[cls]
                
                det = {'name': name, 'conf': conf, 'bbox': [int(x1), int(y1), int(x2), int(y2)],
                       'cx': int((x1+x2)/2), 'cy': int((y1+y2)/2), 'w': int(x2-x1), 'h': int(y2-y1)}
                detections.append(det)
                
                if name == 'ball':
                    if self.ball_target is None or conf > self.ball_target['conf']:
                        self.ball_target = det
                elif name == 'goalpost':
                    if self.goal_target is None or conf > self.goal_target['conf']:
                        self.goal_target = det

        # Draw Debug
        debug_img = frame.copy()
        for det in detections:
            color = (0, 255, 0) if det['name'] == 'ball' else (255, 165, 0)
            cv2.rectangle(debug_img, (det['bbox'][0], det['bbox'][1]), (det['bbox'][2], det['bbox'][3]), color, 2)
            cv2.putText(debug_img, f"{det['name']} {det['conf']:.2f}", (det['bbox'][0], det['bbox'][1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- STATE MACHINE LOGIC ---
        lx, ly, az = 0.0, 0.0, 0.0
        
        if self.state == 'SEARCH_BALL':
            if self.ball_target:
                self.set_state('APPROACH_BALL')
            else:
                # Scan Camera
                self.servo_x += 20 * self.scan_direction
                if self.servo_x >= 2400 or self.servo_x <= 600:
                    self.scan_direction *= -1
                direct_servo_set(0.1, [[1, 2250], [2, self.servo_x]])
                self.set_velocity(0.0, 0.0, 0.0)

        elif self.state == 'APPROACH_BALL':
            if not self.ball_target:
                self.lost_ball_count += 1
                if self.lost_ball_count > 30:
                    self.set_state('SEARCH_BALL')
            else:
                self.lost_ball_count = 0
                dist = self.distance_estimator.estimate_distance(max(self.ball_target['w'], self.ball_target['h'])/2)
                
                # Center camera on ball
                self.servo_x_pid.SetPoint = w/2
                self.servo_x_pid.update(self.ball_target['cx'])
                self.servo_x = int(np.clip(self.servo_x + self.servo_x_pid.output, 500, 2500))
                
                self.servo_y_pid.SetPoint = h/2
                self.servo_y_pid.update(self.ball_target['cy'])
                self.servo_y = int(np.clip(self.servo_y - self.servo_y_pid.output, 1800, 2500))
                direct_servo_set(0.1, [[1, self.servo_y], [2, self.servo_x]])

                # Move Chassis
                servo_err = self.servo_x - 1500
                az = -0.015 * servo_err # Simple P for rotation
                
                if dist > 25:
                    lx = 0.25
                else:
                    lx = 0.0
                    self.set_state('SEARCH_GOAL')
                self.set_velocity(lx, 0.0, az)

        elif self.state == 'SEARCH_GOAL':
            # Keep ball in view while scanning for goal
            if not self.ball_target:
                self.set_state('SEARCH_BALL')
                return
                
            if self.goal_target:
                self.set_state('POSITIONING')
            else:
                # Slowly rotate to find the goal while keeping ball centered
                # PID will handle keeping ball centered
                self.set_velocity(0.0, 0.0, 0.3)
                
                # Keep camera on ball
                self.servo_x_pid.SetPoint = w/2
                self.servo_x_pid.update(self.ball_target['cx'])
                self.servo_x = int(np.clip(self.servo_x + self.servo_x_pid.output, 500, 2500))
                direct_servo_set(0.1, [[1, 2500], [2, self.servo_x]])
                
                if time.time() - self.state_start_time > 15: # Timeout if no goal found
                    self.set_state('APPROACH_BALL') # Try getting closer/further

        elif self.state == 'POSITIONING':
            if not self.ball_target:
                self.set_state('SEARCH_BALL')
                return
            
            if not self.goal_target:
                self.set_state('SEARCH_GOAL')
                return

            # Alignment logic: Robot-Ball-Goal should be a line.
            # In the image frame, ball_x and goal_x should be the same.
            
            ball_x = self.ball_target['cx']
            goal_x = self.goal_target['cx']
            
            # Goal error relative to ball
            # If goal_x > ball_x, goal is to the right of the ball
            # To fix this, we need to move ball RIGHT in the frame
            # To move objects right in the frame, robot must move LEFT (ly > 0)
            error = goal_x - ball_x
            
            # Distance from ball
            dist = self.distance_estimator.estimate_distance(max(self.ball_target['w'], self.ball_target['h'])/2)

            if abs(error) < 40 and abs(self.servo_x - 1500) < 100:
                # Aligned and facing forward
                self.set_state('STRIKE')
            else:
                # PID for strafing alignment
                ly = 0.005 * error # Simple P
                ly = np.clip(ly, -0.3, 0.3)
                
                # Also need to keep the robot facing the ball/goal
                # Use servo_x to align chassis heading
                servo_err = self.servo_x - 1500
                az = -0.015 * servo_err
                
                # Maintain distance
                lx = 0.0
                if dist > 20: lx = 0.1
                elif dist < 15: lx = -0.1
                
                self.set_velocity(lx, ly, az)

        elif self.state == 'STRIKE':
            # Ram the ball!
            self.set_velocity(0.6, 0.0, 0.0)
            if time.time() - self.state_start_time > 1.5:
                self.set_velocity(0.0, 0.0, 0.0)
                self.set_state('SEARCH_BALL')

        # Debug Info
        cv2.putText(debug_img, f"STATE: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out = Image()
        out.height, out.width = h, w
        out.encoding = "bgr8"
        out.step = w * 3
        out.data = debug_img.tobytes()
        self.debug_pub.publish(out)

def main():
    rclpy.init()
    node = AutonomousSoccerStriker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.set_velocity(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
