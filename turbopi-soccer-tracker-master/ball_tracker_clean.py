#!/usr/bin/env python3
"""
Clean Ball Tracker - HiWonder TurboPi
======================================
Uses YOLO model for ball detection with our manually tuned settings.

TUNED SETTINGS (from turbopi_diagnostic.py):
- Servo 1 = TILT: 1200=up, 2250=45deg, 2500=down
- Servo 2 = PAN: 500=right, 1500=center, 2500=left
- Motors: Left(1,3) inverted, Right(2,4) normal

Author: Capstone Team 7
"""
import cv2
import numpy as np
import sys
import serial
import struct
import subprocess
import time

# ============ MODE SETTINGS ============
CAMERA_ONLY_MODE = True  # Set to False to enable chassis movement

# ============ TUNED SETTINGS FROM MANUAL TESTING ============
# All values tested and confirmed in turbopi_diagnostic.py

SERVO_TILT_UP = 1200      # Looking at sky
SERVO_TILT_45 = 2250      # Ideal ball tracking angle
SERVO_TILT_DOWN = 2500    # Looking at floor (max)
SERVO_PAN_RIGHT = 500     # Camera looking right
SERVO_PAN_CENTER = 1500   # Camera centered
SERVO_PAN_LEFT = 2500     # Camera looking left

# Motor movement patterns (all tested)
MOTOR_SPEED = 42  # Tuned speed for smooth movement

# 90 degree alignment (when ball at camera edge)
ALIGN_RIGHT = {'steps': 19, 'speed': 42, 'time': 0.089}
ALIGN_LEFT = {'steps': 19, 'speed': 42, 'time': 0.086}

# 180 degree turn (search behind)
TURN_180 = {'steps': 36, 'speed': 42, 'time': 0.086, 'delay': 0.02}

# 90 degree spin (for 360 search)
SPIN_90 = {'steps': 17, 'speed': 42, 'time': 0.094, 'delay': 0.02}

# ============ SERIAL COMMUNICATION ============
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
        print(f"[WARN] Serial init failed: {e}")
        return False

def checksum_crc8(data):
    check = 0
    for b in data:
        check = CRC8_TABLE[check ^ b]
    return check & 0x00FF

def send_command(func, data):
    global SERIAL_PORT
    if SERIAL_PORT is None:
        return
    buf = [0xAA, 0x55, int(func), len(data)]
    buf.extend(data)
    buf.append(checksum_crc8(bytes(buf[2:])))
    try:
        SERIAL_PORT.write(bytes(buf))
    except:
        pass

def set_servo(tilt, pan, duration=0.15):
    """Set servo positions. tilt=servo1, pan=servo2"""
    duration_ms = int(duration * 1000)
    data = [0x01, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, 2]
    data.extend(struct.pack("<BH", 1, int(tilt)))  # Servo 1 = tilt
    data.extend(struct.pack("<BH", 2, int(pan)))   # Servo 2 = pan
    send_command(4, data)

def set_motors(m1, m2, m3, m4):
    """Set motor speeds. Left(1,3) inverted, Right(2,4) normal."""
    data = [0x05, 4]
    for i, speed in enumerate([m1, m2, m3, m4]):
        data.extend(struct.pack("<Bf", i, float(speed)))
    send_command(3, data)

def stop_motors():
    set_motors(0, 0, 0, 0)

def spin_left(speed=MOTOR_SPEED):
    """Spin counter-clockwise (all positive)"""
    set_motors(speed, speed, speed, speed)

def spin_right(speed=MOTOR_SPEED):
    """Spin clockwise (all negative)"""
    set_motors(-speed, -speed, -speed, -speed)

# ============ MOVEMENT FUNCTIONS (TUNED) ============
def do_90_spin(direction="right"):
    """90 degree pure spin - TUNED"""
    for _ in range(SPIN_90['steps']):
        if direction == "right":
            spin_right(SPIN_90['speed'])
        else:
            spin_left(SPIN_90['speed'])
        time.sleep(SPIN_90['time'])
        stop_motors()
        time.sleep(SPIN_90['delay'])
    stop_motors()

def do_360_search(tilt=SERVO_TILT_45):
    """Full 360 search - scan 4 directions"""
    print("[SEARCH] Starting 360 degree search...")
    for i in range(4):
        # Scan camera left-right
        for pan in range(SERVO_PAN_CENTER, SERVO_PAN_RIGHT, -100):
            set_servo(tilt, pan)
            time.sleep(0.15)
        for pan in range(SERVO_PAN_RIGHT, SERVO_PAN_LEFT, 100):
            set_servo(tilt, pan)
            time.sleep(0.15)
        set_servo(tilt, SERVO_PAN_CENTER)
        time.sleep(0.3)

        if i < 3:  # Don't spin after last scan
            do_90_spin("right")
            time.sleep(0.2)

    print("[SEARCH] 360 search complete")

# ============ ROS2 SETUP ============
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[ERROR] ultralytics not installed!")

# Simple PID class
class PID:
    def __init__(self, P=0.1, I=0.001, D=0.008):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.target = 0
        self.error = 0
        self.last_error = 0
        self.integral = 0

    def update(self, current):
        self.error = self.target - current
        self.integral += self.error
        self.integral = max(-500, min(500, self.integral))  # Anti-windup
        derivative = self.error - self.last_error
        self.last_error = self.error
        return self.Kp * self.error + self.Ki * self.integral + self.Kd * derivative

class BallTracker(Node):
    def __init__(self):
        super().__init__('ball_tracker_clean')

        # YOLO model
        self.model = None
        if YOLO_OK:
            try:
                self.model = YOLO('/home/ubuntu/best.pt')
                self.get_logger().info("YOLO model loaded!")
            except Exception as e:
                self.get_logger().error(f"YOLO load failed: {e}")

        # Publishers
        self.debug_pub = self.create_publisher(Image, '/yolo_debug', 10)

        # Subscriber
        self.create_subscription(Image, '/image_raw', self.on_image, 1)

        # Servo positions
        self.pan = SERVO_PAN_CENTER
        self.tilt = SERVO_TILT_45

        # PID controllers (HiWonder-style)
        self.pan_pid = PID(P=0.25, I=0.05, D=0.009)

        # Tracking state
        self.lost_frames = 0
        self.frame_count = 0
        self.scan_direction = 1

        # Initialize camera position
        set_servo(self.tilt, self.pan, 1.0)
        self.get_logger().info("=" * 50)
        self.get_logger().info("Ball Tracker Ready!")
        self.get_logger().info(f"Mode: {'CAMERA ONLY' if CAMERA_ONLY_MODE else 'FULL'}")
        self.get_logger().info("=" * 50)

    def on_image(self, msg):
        try:
            # Convert ROS image to OpenCV
            if "yuy" in msg.encoding.lower():
                frame = cv2.cvtColor(
                    np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 2),
                    cv2.COLOR_YUV2BGR_YUYV
                )
            else:
                frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

            h, w = frame.shape[:2]
            result = frame.copy()
            self.frame_count += 1

            # Log every 30 frames
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Frame {self.frame_count}: {w}x{h}")

            # ============ YOLO DETECTION ============
            ball = None
            if self.model:
                detections = self.model(frame, conf=0.35, verbose=False)

                if detections[0].boxes is not None:
                    for box in detections[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # Get class name
                        cls_name = self.model.names[cls_id] if cls_id < len(self.model.names) else f"cls_{cls_id}"

                        # Skip small detections
                        if (x2 - x1) < 25 or (y2 - y1) < 25:
                            continue

                        # Calculate center and radius
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        radius = max(x2 - x1, y2 - y1) // 2

                        # Draw all detections
                        color = (0, 255, 0) if cls_name == 'ball' else (0, 165, 255)

                        # Draw circle outline (HiWonder style)
                        cv2.circle(result, (cx, cy), radius, color, 2)
                        cv2.circle(result, (cx, cy), 5, color, -1)  # Center dot

                        # Draw label
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(result, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Track ball (highest confidence)
                        if cls_name == 'ball':
                            if ball is None or conf > ball['conf']:
                                ball = {'cx': cx, 'cy': cy, 'radius': radius, 'conf': conf}

            # ============ BALL TRACKING ============
            if ball:
                self.lost_frames = 0
                cx, cy = ball['cx'], ball['cy']

                # PID tracking - keep ball centered
                self.pan_pid.target = w // 2
                pan_adjust = self.pan_pid.update(cx)

                # Update servo (HiWonder adds adjustment)
                self.pan = int(np.clip(self.pan + pan_adjust, SERVO_PAN_RIGHT, SERVO_PAN_LEFT))
                set_servo(self.tilt, self.pan)

                # Status
                status = f"TRACKING: r={ball['radius']} pan={self.pan}"
                cv2.putText(result, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.frame_count % 30 == 0:
                    self.get_logger().info(f"TRACKING ball: pos=({cx},{cy}) r={ball['radius']} pan={self.pan}")

            else:
                # Ball not found
                self.lost_frames += 1

                if self.lost_frames > 30:
                    # Camera scan
                    self.pan += 20 * self.scan_direction
                    if self.pan >= 2200:
                        self.scan_direction = -1
                        self.pan = 2200
                    elif self.pan <= 800:
                        self.scan_direction = 1
                        self.pan = 800
                    set_servo(self.tilt, self.pan)

                    status = f"SEARCHING... pan={self.pan}"
                else:
                    status = f"Lost ({self.lost_frames})"

                cv2.putText(result, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                # 360 search after long loss (only if chassis enabled)
                if self.lost_frames == 150 and not CAMERA_ONLY_MODE:
                    do_360_search(self.tilt)
                    self.lost_frames = 0

            # Draw crosshair
            cv2.line(result, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 255), 1)
            cv2.line(result, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 255), 1)

            # Draw servo info
            cv2.putText(result, f"pan={self.pan} tilt={self.tilt}",
                       (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Publish debug image
            out = Image()
            out.header = msg.header
            out.height, out.width = h, w
            out.encoding = "bgr8"
            out.step = w * 3
            out.data = result.tobytes()
            self.debug_pub.publish(out)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    # Kill conflicting processes
    for proc in ['simple_ball', 'ball_track', '/app/tracking']:
        subprocess.run(['pkill', '-f', proc], capture_output=True)
    time.sleep(0.3)

    # Initialize serial
    init_serial()

    # ROS2
    rclpy.init()
    node = BallTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_motors()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
