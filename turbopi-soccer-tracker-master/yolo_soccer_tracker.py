#!/usr/bin/env python3
"""
YOLO Soccer Tracker - HiWonder TurboPi
=======================================
Uses YOLOv8 model (best.pt) to detect:
  - ball (primary tracking target)
  - goalpost (for navigation)
  - Robot-car (opponent detection)

Features:
  - Camera pan/tilt servo tracking (PID control)
  - Multi-object detection visualization
  - Prioritizes ball tracking
  - Shows all detected objects with different colors

Model: best.pt (YOLOv8, trained on soccer objects)
Author: Capstone Team 7
"""
import cv2
import math
import numpy as np
import sys
import serial
import struct
import subprocess
import rclpy

def kill_conflicting_processes():
    """Kill ball tracking scripts that might conflict."""
    conflicts = ['simple_ball', 'ball_track', '/app/tracking']
    for name in conflicts:
        try:
            subprocess.run(['pkill', '-f', name], capture_output=True)
        except:
            pass
    import time
    time.sleep(0.3)
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
import sdk.pid as pid

# ============ BALL DISTANCE ESTIMATION ============
class BallDistanceEstimator:
    """
    Estimate ball distance using monocular pinhole camera model.

    Formula: distance = (real_diameter × focal_length) / diameter_pixels

    Calibration: Place ball at known distance, note radius in pixels,
    then calculate focal_length = (diameter_pixels × distance) / real_diameter
    """

    def __init__(self, focal_length_pixels=450.0, ball_diameter_cm=6.5):
        """
        Args:
            focal_length_pixels: Camera focal length (calibrate with known distance)
            ball_diameter_cm: Real ball diameter in cm (measure your ball)
        """
        self.focal_length = focal_length_pixels
        self.ball_diameter_cm = ball_diameter_cm

        # Smoothing for stable distance readings
        self.last_distance = None
        self.smooth_alpha = 0.3  # Lower = more smoothing

    def estimate_distance(self, ball_radius_pixels):
        """
        Estimate distance to ball.

        Args:
            ball_radius_pixels: Ball radius from YOLO detection

        Returns:
            distance_cm: Estimated distance in centimeters (smoothed)
        """
        if ball_radius_pixels <= 0:
            return None

        ball_diameter_pixels = ball_radius_pixels * 2
        distance_cm = (self.ball_diameter_cm * self.focal_length) / ball_diameter_pixels

        # Smooth the reading
        if self.last_distance is not None:
            distance_cm = self.smooth_alpha * distance_cm + (1 - self.smooth_alpha) * self.last_distance
        self.last_distance = distance_cm

        return distance_cm

    def get_distance_zone(self, distance_cm):
        """
        Classify distance into zones for speed control.

        Returns: (zone_name, recommended_speed)
        """
        if distance_cm is None:
            return ("UNKNOWN", 0.0)
        elif distance_cm < 15:
            return ("VERY_CLOSE", 0.0)   # Stop - almost touching
        elif distance_cm < 25:
            return ("CLOSE", 0.10)       # Very slow approach
        elif distance_cm < 40:
            return ("MEDIUM", 0.20)      # Normal approach
        elif distance_cm < 60:
            return ("FAR", 0.30)         # Faster pursuit
        else:
            return ("VERY_FAR", 0.40)    # Fast pursuit

    def calibrate_from_measurement(self, radius_pixels, known_distance_cm):
        """
        Calibrate focal length from a known measurement.
        Place ball at known distance and call this with measured radius.
        """
        diameter_pixels = radius_pixels * 2
        self.focal_length = (diameter_pixels * known_distance_cm) / self.ball_diameter_cm
        return self.focal_length

# ============ DIRECT SERIAL CONTROL (TESTED AND WORKING) ============
# SERVO MAPPING (confirmed through testing):
#   Servo 1 = TILT (vertical): 1000=UP (sky), 2500=DOWN (floor)
#   Servo 2 = PAN (horizontal): 500=RIGHT, 1500=CENTER, 2500=LEFT
#
# MOTOR MAPPING (confirmed through individual wheel testing):
#   Motor 1 = Front-Left:  negative=forward, positive=backward
#   Motor 2 = Front-Right: positive=forward, negative=backward
#   Motor 3 = Rear-Left:   negative=forward, positive=backward
#   Motor 4 = Rear-Right:  positive=forward, negative=backward

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
    """Set servo positions using direct serial.
    TESTED: Matches turbopi_diagnostic.py exactly.

    positions: list of [servo_id, position] pairs
    servo_id: 1=TILT (1200=UP, 2250=45deg, 2500=DOWN), 2=PAN (500=R, 1500=C, 2500=L)
    """
    duration_ms = int(duration * 1000)
    data = [0x01, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, len(positions)]
    for servo_id, pos in positions:
        data.extend(struct.pack("<BH", servo_id, pos))
    buf_write(4, data)

def direct_motor_set(motor_speeds):
    """Set motor speeds using direct serial (TESTED AND WORKING)
    motor_speeds: list of [motor_id, duty] pairs

    MOTOR MAPPING (tested individually):
    - Motor 1 = Front-Left: NEGATIVE = forward, POSITIVE = backward
    - Motor 2 = Front-Right: POSITIVE = forward, NEGATIVE = backward
    - Motor 3 = Rear-Left: NEGATIVE = forward, POSITIVE = backward
    - Motor 4 = Rear-Right: POSITIVE = forward, NEGATIVE = backward

    For FORWARD motion: M1=-speed, M2=+speed, M3=-speed, M4=+speed
    For BACKWARD motion: M1=+speed, M2=-speed, M3=+speed, M4=-speed
    """
    data = [0x05, len(motor_speeds)]  # 0x05 = duty command
    for motor_id, duty in motor_speeds:
        data.extend(struct.pack("<Bf", motor_id - 1, float(duty)))
    buf_write(3, data)  # function 3 = motor

def mecanum_drive(linear_x, linear_y, angular_z, max_duty=80):
    """
    Convert velocity commands to mecanum wheel speeds using direct serial.

    Args:
        linear_x: Forward/backward speed (-1.0 to 1.0), positive = forward
        linear_y: Left/right strafe speed (-1.0 to 1.0), positive = left
        angular_z: Rotation speed (-1.0 to 1.0), positive = counter-clockwise
        max_duty: Maximum motor duty cycle (0-100)
    """
    # Scale inputs to max_duty
    vx = linear_x * max_duty
    vy = linear_y * max_duty
    wz = angular_z * max_duty

    # Mecanum kinematics (standard X-configuration)
    # FL, FR, RL, RR
    fl = vx - vy - wz  # Front-Left
    fr = vx + vy + wz  # Front-Right
    rl = vx + vy - wz  # Rear-Left
    rr = vx - vy + wz  # Rear-Right

    # Apply motor direction corrections (from testing):
    # Motors 1 and 3 (left side) have reversed direction
    m1 = -fl  # Front-Left: negate for correct direction
    m2 = fr   # Front-Right: already correct
    m3 = -rl  # Rear-Left: negate for correct direction
    m4 = rr   # Rear-Right: already correct

    # Clip to max duty
    m1 = max(-max_duty, min(max_duty, m1))
    m2 = max(-max_duty, min(max_duty, m2))
    m3 = max(-max_duty, min(max_duty, m3))
    m4 = max(-max_duty, min(max_duty, m4))

    direct_motor_set([[1, m1], [2, m2], [3, m3], [4, m4]])

# Kill conflicting processes and initialize serial
print("[INIT] Killing conflicting processes...")
kill_conflicting_processes()
USE_DIRECT_SERIAL = init_serial()

# ============ MODE SETTINGS ============
# Set to True for camera-only tracking (no chassis movement)
# Useful for testing camera tracking before enabling robot movement
CAMERA_ONLY_MODE = True  # Set to False to enable chassis movement

# ============ TUNED MOVEMENT SETTINGS (from extensive testing) ============
# These values were tuned through physical testing on the robot
# See turbopi_diagnostic.py for full documentation

TUNED_MOVEMENTS = {
    # 90 DEGREE BALL ALIGNMENT (camera + chassis sync)
    # When camera sees ball on far edge, rotate chassis to face it
    'align_right': {'speed': 42, 'time': 0.089, 'steps': 19},  # Pan 600->1500
    'align_left': {'speed': 42, 'time': 0.086, 'steps': 19},   # Pan 2400->1500

    # 180 DEGREE TURN (search mode - ball behind)
    'turn_180': {'speed': 42, 'time': 0.086, 'steps': 36, 'delay': 0.02},

    # 90 DEGREE PURE SPIN (for 360 search - no camera tracking)
    'spin_90': {'speed': 42, 'time': 0.094, 'steps': 17, 'delay': 0.02},

    # CAMERA SCAN settings
    'scan': {'pan_min': 600, 'pan_max': 2400, 'pan_center': 1500, 'tilt': 2250},
}

def motor_stop():
    """Stop all motors immediately."""
    direct_motor_set([[1, 0], [2, 0], [3, 0], [4, 0]])

def align_to_ball(direction, tilt=2250):
    """
    Align car body to where camera is looking (90 degree rotation).
    TUNED through extensive testing.

    When camera sees ball on far right/left edge, this rotates the
    chassis so the front faces that direction, while camera returns
    to center.

    Args:
        direction: "right" or "left" - where the ball is
        tilt: camera tilt angle (default 2250 = 45 degrees down)
    """
    import time

    if direction == "right":
        # Camera looks far right, then chassis rotates right while camera centers
        direct_servo_set(0.5, [[1, tilt], [2, 600]])  # Look far right
        time.sleep(0.6)

        # TUNED: speed=42, time=0.089s for 90 degree right rotation
        for pan in range(600, 1550, 50):
            direct_motor_set([[1, -42], [2, -42], [3, -42], [4, -42]])  # Spin right
            time.sleep(0.089)
            motor_stop()
            direct_servo_set(0.15, [[1, tilt], [2, pan]])
            time.sleep(0.08)
    else:
        # Camera looks far left, then chassis rotates left while camera centers
        direct_servo_set(0.5, [[1, tilt], [2, 2400]])  # Look far left
        time.sleep(0.6)

        # TUNED: speed=42, time=0.086s for 90 degree left rotation
        for pan in range(2400, 1450, -50):
            direct_motor_set([[1, 42], [2, 42], [3, 42], [4, 42]])  # Spin left
            time.sleep(0.086)
            motor_stop()
            direct_servo_set(0.15, [[1, tilt], [2, pan]])
            time.sleep(0.08)

    # Final center
    direct_servo_set(0.3, [[1, tilt], [2, 1500]])
    time.sleep(0.4)
    motor_stop()

def turn_180(direction="left", tilt=2250):
    """
    Turn 180 degrees when ball is behind (not visible).
    TUNED through testing.

    Args:
        direction: "left" or "right" - which way to spin
        tilt: camera tilt angle
    """
    import time

    # Keep camera centered at 45 degrees
    direct_servo_set(0.3, [[1, tilt], [2, 1500]])
    time.sleep(0.3)

    # TUNED: 36 steps, speed=42, time=0.086s
    for i in range(36):
        if direction == "left":
            direct_motor_set([[1, 42], [2, 42], [3, 42], [4, 42]])  # All positive = spin left
        else:
            direct_motor_set([[1, -42], [2, -42], [3, -42], [4, -42]])  # All negative = spin right
        time.sleep(0.086)
        motor_stop()
        time.sleep(0.02)

    motor_stop()

def spin_90(direction="right"):
    """
    Pure 90 degree spin (no camera tracking).
    TUNED for 360 search mode.

    Args:
        direction: "right" or "left"
    """
    import time

    # TUNED: 17 steps, speed=42, time=0.094s
    for i in range(17):
        if direction == "right":
            direct_motor_set([[1, -42], [2, -42], [3, -42], [4, -42]])
        else:
            direct_motor_set([[1, 42], [2, 42], [3, 42], [4, 42]])
        time.sleep(0.094)
        motor_stop()
        time.sleep(0.02)

    motor_stop()

def scan_camera_area(tilt=2250):
    """
    Scan camera left to right at 45 degrees.
    Used in 360 search mode.

    Returns True if this is where ball detection would trigger.
    (In actual use, the main tracking loop handles detection)
    """
    import time

    direct_servo_set(0.3, [[1, tilt], [2, 1500]])
    time.sleep(0.3)

    # Scan right
    for pan in range(1500, 600, -100):
        direct_servo_set(0.15, [[1, tilt], [2, pan]])
        time.sleep(0.15)

    # Scan left
    for pan in range(600, 2400, 100):
        direct_servo_set(0.15, [[1, tilt], [2, pan]])
        time.sleep(0.15)

    # Back to center
    direct_servo_set(0.3, [[1, tilt], [2, 1500]])
    time.sleep(0.3)

def search_360(tilt=2250):
    """
    Full 360 degree ball search.
    TUNED through testing.

    Scans in 4 directions (front, right, back, left)
    then returns to start position.

    Use when ball is not visible for extended time.
    """
    import time

    # Direction 1: FRONT (0 degrees)
    scan_camera_area(tilt)
    time.sleep(0.3)

    # Direction 2: RIGHT (90 degrees)
    spin_90("right")
    time.sleep(0.2)
    scan_camera_area(tilt)
    time.sleep(0.3)

    # Direction 3: BACK (180 degrees)
    spin_90("right")
    time.sleep(0.2)
    scan_camera_area(tilt)
    time.sleep(0.3)

    # Direction 4: LEFT (270 degrees)
    spin_90("right")
    time.sleep(0.2)
    scan_camera_area(tilt)
    time.sleep(0.3)

    # Return to original direction
    spin_90("right")

    motor_stop()
    direct_servo_set(0.3, [[1, tilt], [2, 1500]])

# Fallback to SDK if serial fails
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/driver/sdk/sdk")
try:
    from ros_robot_controller_sdk import Board
    BOARD = Board()
    USE_SDK = True
except:
    BOARD = None
    USE_SDK = False

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")


def is_circular_shape(width, height, min_aspect_ratio=0.75):
    """
    Check if bounding box is roughly square (circular object).
    Balls should have aspect ratio close to 1.0
    Returns True if object is circular enough.
    """
    if height == 0 or width == 0:
        return False
    aspect_ratio = min(width, height) / max(width, height)
    return aspect_ratio >= min_aspect_ratio


def verify_ball_shape(frame, x1, y1, x2, y2, min_circularity=0.7):
    """
    Verify detected object is actually round like a ball.
    Uses contour analysis to check circularity.
    Returns (is_valid, circularity_score)
    """
    try:
        # Extract ROI
        roi = frame[max(0,y1):y2, max(0,x1):x2]
        if roi.size == 0:
            return False, 0.0

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return True, 0.8  # No contours, trust YOLO

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < 50:  # Too small
            return True, 0.8  # Trust YOLO for small objects

        # Calculate circularity: 4*pi*area / perimeter^2
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return True, 0.8

        circularity = (4 * math.pi * area) / (perimeter * perimeter)

        return circularity >= min_circularity, circularity

    except Exception:
        return True, 0.8  # On error, trust YOLO


class YOLOSoccerTracker(Node):
    def __init__(self):
        super().__init__('yolo_soccer_tracker')

        # ==================== MODEL CONFIGURATION ====================
        self.model_path = '/home/ubuntu/best.pt'  # Path inside Docker
        self.conf_threshold = 0.45
        self.iou_threshold = 0.5

        # Minimum detection size (pixels) - filters out UI elements/noise
        self.min_detection_size = 30  # Reject detections smaller than 30x30

        # Higher confidence thresholds for specific classes (reduce false positives)
        self.class_conf_thresholds = {
            'Robot-car': 0.70,  # Require high confidence for robot detection
            'ball': 0.45,       # Normal threshold for ball
            'goalpost': 0.50,   # Normal threshold for goal
        }

        # Class names from your trained model
        self.class_names = ['Robot-car', 'ball', 'goalpost']

        # Colors for each class (BGR format)
        self.class_colors = {
            'Robot-car': (0, 0, 255),    # Red - opponent robots
            'ball': (0, 255, 0),          # Green - primary target
            'goalpost': (255, 165, 0),    # Orange - goal
        }

        # Primary tracking target
        self.primary_target = 'ball'

        # ==================== LOAD YOLO MODEL ====================
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
                self.get_logger().info(f"Loaded YOLO model: {self.model_path}")
                self.get_logger().info(f"Classes: {self.class_names}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        else:
            self.get_logger().error("YOLO not available - install ultralytics")

        # ==================== ROS PUBLISHERS ====================
        self.servo_pub = self.create_publisher(
            SetPWMServoState,
            '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(
            Image, '/yolo_debug', 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # ROS Subscriber
        self.create_subscription(Image, '/image_raw', self.image_callback, 1)

        # ==================== SERVO CONFIGURATION ====================
        # PID Controllers - HiWonder tuned values for STABLE centering
        # Research: HiWonder uses P=0.1-0.145, I=0.001-0.02, D=0.0007-0.008
        # Lower gains = smoother tracking, no oscillation
        self.servo_x_pid = pid.PID(P=0.20, I=0.008, D=0.008)   # Pan - stable
        self.servo_y_pid = pid.PID(P=0.40, I=0.015, D=0.015)  # Tilt - stable, no oscillation

        # SERVO VALUES (tested on this robot):
        # Servo 1 = TILT: 1000=UP (sky), 2500=DOWN (floor)
        # Servo 2 = PAN: left/right, center 1500
        self.servo_x = 1500  # Pan - center
        self.servo_y = 2500  # Tilt - max down (looking at floor)

        # Servo limits (tested) - expanded tilt for better centering
        self.servo_min_x, self.servo_max_x = 500, 2500   # Pan range
        self.servo_min_y, self.servo_max_y = 1800, 2500  # Tilt range (expanded up for precision)

        # Dead zone - small to prevent oscillation while staying accurate
        self.deadzone_x = 8   # Horizontal - balanced
        self.deadzone_y = 5   # Vertical - small to prevent up/down oscillation

        # PID output clamping - prevent over-correction
        self.max_pid_output = 30  # Max servo change per frame (slightly higher)

        # ==================== CHASSIS CONTROL ====================
        self.chassis_enabled = not CAMERA_ONLY_MODE  # Respects global mode setting
        self.target_radius = 80  # Desired ball size in pixels (larger = closer)
        self.radius_deadzone = 10  # +/- pixels tolerance
        self.servo_center = 1500  # Servo center position
        self.align_threshold = 50  # Servo deviation threshold for "aligned"

        # Chassis PIDs - EXACT values from working ball_tracker_improved.py
        # CRITICAL: D must be 10×P for rotation to prevent oscillation
        self.chassis_yaw_pid = pid.PID(P=0.012, I=0.002, D=0.08)   # Rotation - increased P
        self.chassis_dist_pid = pid.PID(P=0.015, I=0.003, D=0.05)  # Distance - increased for faster approach
        self.min_approach_speed = 0.25  # Minimum forward speed (increased)

        # Velocity smoothing - CRITICAL for stability
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0

        # ==================== TRACKING STATE ====================
        self.lost_count = 0
        self.scan_direction = 1  # Start at center, go LEFT first (+), then RIGHT (-)
        self.search_speed = 25  # Faster scan for visibility

        # Smoothing
        self.smoothing_alpha = 0.3
        self.last_x = None
        self.last_y = None
        self.last_servo_x = 1500
        self.last_servo_y = 2500

        # Detection storage
        self.current_detections = []
        self.frame_count = 0

        # TEMPORAL FILTERING - require consistent detections
        self.detection_history = []  # List of recent ball detections
        self.history_length = 3      # Number of frames to track
        self.min_consistent_frames = 2  # Require 2+ frames to confirm
        self.confirmed_target = None    # Last confirmed target

        # STATE MACHINE for align-then-approach
        # States: 'SEARCHING', 'ALIGNING', 'APPROACHING', 'HOLDING',
        #         'ALIGNING_CHASSIS', 'SEARCHING_360'
        self.tracking_state = 'SEARCHING'
        self.state_frames = 0  # Frames in current state

        # Chassis alignment thresholds (tuned values)
        self.far_edge_pan = 200       # If pan within 200 of limits, trigger chassis align
        self.chassis_align_cooldown = 0  # Frames to wait before another alignment
        self.search_360_threshold = 150  # Lost frames before 360 search (~5 seconds)
        self.last_180_direction = "left" # Alternate 180 turn direction

        # STABILITY: Prevent search from triggering on brief detection gaps
        self.last_seen_frame = 0      # Frame number when ball was last seen
        self.min_lost_before_search = 30  # Must be lost for 30+ frames to start scanning

        # Adaptive confidence based on state
        self.scanning_confidence = 0.55  # Higher during scanning to avoid false positives
        self.tracking_confidence = 0.40  # Lower when already tracking (trust history)

        # ==================== DISTANCE ESTIMATION ====================
        # Pinhole camera model: distance = (real_size × focal_length) / pixel_size
        # Calibration: focal_length=450 works for ~6.5cm ball at typical distances
        # Adjust focal_length if distance readings are off
        self.distance_estimator = BallDistanceEstimator(
            focal_length_pixels=450.0,  # Calibrate: increase if reading too close
            ball_diameter_cm=6.5        # Measure your actual ball diameter
        )
        self.current_distance_cm = None
        self.current_distance_zone = "UNKNOWN"

        # Initialize servo position - FORCE camera DOWN on startup
        # Use DIRECT SERIAL (proven to work)
        if USE_DIRECT_SERIAL:
            # Servo 1 = TILT (up/down), Servo 2 = PAN (left/right)
            # Tilt: 1000=UP, 2500=DOWN
            direct_servo_set(1.0, [[1, self.servo_y], [2, self.servo_x]])
            self.get_logger().info(f"INIT (direct serial): tilt={self.servo_y} pan={self.servo_x}")
        elif USE_SDK and BOARD is not None:
            BOARD.pwm_servo_set_position(1.0, [[1, self.servo_y], [2, self.servo_x]])
            self.get_logger().info(f"INIT (SDK): tilt={self.servo_y} pan={self.servo_x}")
        self._last_pub_x = self.servo_x
        self._last_pub_y = self.servo_y

        self.get_logger().info("=" * 50)
        self.get_logger().info("YOLO Soccer Tracker Ready!")
        self.get_logger().info(f"Tracking: {self.primary_target}")
        self.get_logger().info(f"Detecting: {self.class_names}")
        self.get_logger().info("=" * 50)

    def publish_servo(self, servo_x, servo_y):
        """Publish servo positions using SDK directly (more reliable)"""
        servo_x = int(np.clip(servo_x, self.servo_min_x, self.servo_max_x))
        servo_y = int(np.clip(servo_y, self.servo_min_y, self.servo_max_y))

        # ONLY send command if position changed by more than 5 units
        if not hasattr(self, '_last_pub_x'):
            self._last_pub_x = 1500
            self._last_pub_y = 2500  # Match initial DOWN position

        x_changed = abs(servo_x - self._last_pub_x) > 5  # Lower threshold for responsive scanning
        y_changed = abs(servo_y - self._last_pub_y) > 5

        if not x_changed and not y_changed:
            return  # Don't send - prevents jitter!

        # Use DIRECT SERIAL (proven to work)
        if USE_DIRECT_SERIAL:
            try:
                # Servo 1 = TILT (up/down), Servo 2 = PAN (left/right)
                direct_servo_set(0.15, [[1, servo_y], [2, servo_x]])
            except Exception as e:
                self.get_logger().warn(f"SDK servo error: {e}")

        self._last_pub_x = servo_x
        self._last_pub_y = servo_y
        self.servo_x, self.servo_y = servo_x, servo_y

    def stop_chassis(self):
        """Stop chassis movement using direct serial (TESTED AND WORKING)"""
        if USE_DIRECT_SERIAL:
            direct_motor_set([[1, 0], [2, 0], [3, 0], [4, 0]])
        else:
            # Fallback to ROS cmd_vel
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

    def set_velocity(self, linear_x, angular_z):
        """
        Set robot velocity using DIRECT SERIAL (TESTED AND WORKING).
        Bypasses ROS2 mecanum controller for reliable motor control.

        linear_x: forward/backward speed (-1.0 to 1.0), positive = forward
        angular_z: rotation speed (-1.0 to 1.0), positive = counter-clockwise
        """
        # Smooth velocity changes (HiWonder uses alpha=0.7)
        linear_x = self.smooth_value(linear_x, self.last_linear_x, 0.7)
        angular_z = self.smooth_value(angular_z, self.last_angular_z, 0.7)
        self.last_linear_x = linear_x
        self.last_angular_z = angular_z

        # Clip values
        linear_x = float(np.clip(linear_x, -1.0, 1.0))
        angular_z = float(np.clip(angular_z, -1.0, 1.0))

        # Use DIRECT SERIAL for motors (proven to work in testing)
        if USE_DIRECT_SERIAL:
            mecanum_drive(linear_x, 0.0, angular_z, max_duty=80)
        else:
            # Fallback to ROS cmd_vel
            twist = Twist()
            twist.linear.x = linear_x * 0.5  # Scale to m/s
            twist.linear.y = 0.0
            twist.angular.z = angular_z * 1.5  # Scale to rad/s
            self.cmd_vel_pub.publish(twist)

    def smooth_value(self, current, last, alpha=0.4):
        """Exponential moving average smoothing"""
        if last is None:
            return current
        return alpha * current + (1 - alpha) * last

    def detect_objects(self, frame):
        """
        Run YOLO inference on frame
        Returns list of detections with class, bbox, confidence
        """
        if self.model is None:
            return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detections = []

            if results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].item()
                    class_id = int(boxes.cls[i].item())

                    # Get class name
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    # Calculate center and size
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    radius = int(max(width, height) / 2)

                    # FILTER: Skip small detections (UI elements, noise)
                    if width < self.min_detection_size or height < self.min_detection_size:
                        continue  # Skip this detection

                    # FILTER: Apply per-class confidence thresholds
                    min_conf = self.class_conf_thresholds.get(class_name, self.conf_threshold)
                    if confidence < min_conf:
                        continue  # Skip low confidence detections for this class

                    detections.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'cx': cx,
                        'cy': cy,
                        'width': width,
                        'height': height,
                        'radius': radius
                    })

            return detections

        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
            return []

    def get_primary_target(self, detections, frame=None):
        """
        Get the primary tracking target (ball) from detections.
        Uses shape validation to filter false positives.
        """
        primary_detections = [
            d for d in detections
            if d['class_name'] == self.primary_target
        ]

        if not primary_detections:
            return None

        # Apply shape validation to filter non-ball detections
        valid_balls = []
        for det in primary_detections:
            # Check aspect ratio (balls are round, ~1.0 ratio)
            if not is_circular_shape(det['width'], det['height'], min_aspect_ratio=0.6):
                continue  # Skip non-circular detections

            # For high confidence detections, do contour-based circularity check
            if frame is not None and det['confidence'] < 0.7:
                is_valid, circularity = verify_ball_shape(
                    frame, det['x1'], det['y1'], det['x2'], det['y2'],
                    min_circularity=0.55
                )
                if not is_valid:
                    continue  # Skip non-circular shapes
                det['circularity'] = circularity

            valid_balls.append(det)

        if not valid_balls:
            # Fall back to highest confidence if all fail shape check
            return max(primary_detections, key=lambda x: x['confidence'])

        # Return highest confidence valid ball
        return max(valid_balls, key=lambda x: x['confidence'])

    def image_callback(self, msg):
        try:
            # Debug log every 30 frames
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Frame {self.frame_count}: {msg.width}x{msg.height} {msg.encoding}")

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

            # ==================== YOLO DETECTION ====================
            # Adaptive confidence based on tracking state
            if self.tracking_state == 'SEARCHING':
                self.conf_threshold = self.scanning_confidence
            else:
                self.conf_threshold = self.tracking_confidence

            detections = self.detect_objects(frame)
            self.current_detections = detections

            # Get primary target (ball) with shape verification
            raw_target = self.get_primary_target(detections, frame)

            # ==================== TEMPORAL FILTERING ====================
            # Track detection history to filter out single-frame false positives
            self.detection_history.append(raw_target)
            if len(self.detection_history) > self.history_length:
                self.detection_history.pop(0)

            # Count valid detections in history
            valid_frames = sum(1 for t in self.detection_history if t is not None)

            # Require consistent detections to confirm target
            if valid_frames >= self.min_consistent_frames and raw_target is not None:
                target = raw_target
                self.confirmed_target = raw_target
            elif valid_frames >= 1 and self.confirmed_target is not None:
                # Continue tracking if we have a recent confirmed target
                target = raw_target if raw_target else self.confirmed_target
            else:
                target = None
                self.confirmed_target = None

            # ==================== TRACKING LOGIC (FLOOR-ONLY) ====================
            # Ball is ALWAYS on floor - only track horizontally (pan)
            # Tilt stays FIXED at 1200 (looking at floor)

            if target is not None:
                self.lost_count = 0
                self.last_seen_frame = self.frame_count  # Track when we last saw the ball

                cx, cy = target['cx'], target['cy']
                radius = target['radius']

                # ==================== DISTANCE ESTIMATION ====================
                self.current_distance_cm = self.distance_estimator.estimate_distance(radius)
                self.current_distance_zone, recommended_speed = self.distance_estimator.get_distance_zone(
                    self.current_distance_cm
                )

                # ==================== CHASSIS ALIGNMENT CHECK ====================
                # If ball is at far edge of camera view, align chassis to face it
                # This uses the tuned 90-degree alignment from testing
                # ONLY if chassis movement is enabled (not camera-only mode)
                if self.chassis_align_cooldown > 0:
                    self.chassis_align_cooldown -= 1

                if self.chassis_enabled and self.chassis_align_cooldown == 0 and self.tracking_state != 'ALIGNING_CHASSIS':
                    # Check if camera is looking far right or left
                    if self.servo_x <= (self.servo_min_x + self.far_edge_pan):
                        # Ball is on FAR RIGHT - align chassis
                        self.get_logger().info("BALL AT FAR RIGHT - Aligning chassis...")
                        self.tracking_state = 'ALIGNING_CHASSIS'
                        align_to_ball("right", tilt=2250)
                        self.servo_x = 1500  # Camera now centered
                        self.chassis_align_cooldown = 60  # Wait before next align
                        self.tracking_state = 'ALIGNING'
                        self.get_logger().info("Chassis aligned RIGHT - resuming tracking")

                    elif self.servo_x >= (self.servo_max_x - self.far_edge_pan):
                        # Ball is on FAR LEFT - align chassis
                        self.get_logger().info("BALL AT FAR LEFT - Aligning chassis...")
                        self.tracking_state = 'ALIGNING_CHASSIS'
                        align_to_ball("left", tilt=2250)
                        self.servo_x = 1500  # Camera now centered
                        self.chassis_align_cooldown = 60  # Wait before next align
                        self.tracking_state = 'ALIGNING'
                        self.get_logger().info("Chassis aligned LEFT - resuming tracking")

                # Smooth horizontal position
                if self.last_x is not None:
                    ctrl_x = self.smooth_value(cx, self.last_x)
                else:
                    ctrl_x = cx
                self.last_x = ctrl_x

                # Smooth vertical position
                if self.last_y is None:
                    self.last_y = cy
                ctrl_y = self.smooth_value(cy, self.last_y)
                self.last_y = ctrl_y

                # Apply deadzone for both axes
                track_x = w/2 if abs(ctrl_x - w/2) < self.deadzone_x else ctrl_x
                track_y = h/2 if abs(ctrl_y - h/2) < self.deadzone_y else ctrl_y

                # HORIZONTAL PID - track ball left/right
                self.servo_x_pid.SetPoint = w / 2
                self.servo_x_pid.update(track_x)
                pid_x_output = self.servo_x_pid.output

                # VERTICAL PID - track ball up/down
                self.servo_y_pid.SetPoint = h / 2
                self.servo_y_pid.update(track_y)
                pid_y_output = self.servo_y_pid.output

                # CLAMP PID output to prevent over-correction (HiWonder technique)
                pid_x_output = max(-self.max_pid_output, min(self.max_pid_output, pid_x_output))
                pid_y_output = max(-self.max_pid_output, min(self.max_pid_output, pid_y_output))

                # PAN: Add PID output (ball right = servo increases)
                new_servo_x = self.servo_x + int(pid_x_output)

                # TILT: SUBTRACT PID output (ball above center = tilt UP = decrease servo)
                # Servo: 1000=UP, 2500=DOWN. Ball above (y<240) needs tilt UP (lower value)
                new_servo_y = self.servo_y - int(pid_y_output)

                # Light smoothing for pan
                new_servo_x = int(0.9 * new_servo_x + 0.1 * self.last_servo_x)
                self.last_servo_x = new_servo_x
                self.servo_x = new_servo_x

                # No smoothing for tilt - instant response for precision
                self.last_servo_y = new_servo_y
                self.servo_y = new_servo_y

                # Update servo position
                self.publish_servo(self.servo_x, self.servo_y)

                # ==================== CHASSIS CONTROL (ALIGN-THEN-APPROACH) ====================
                # State machine: First ALIGN (rotate only), then APPROACH (straight)
                if self.chassis_enabled:
                    # Ball position error: how far from center of image
                    ball_x_error = abs(cx - w/2)  # Distance from center in pixels
                    # Distance error: positive = ball far, negative = ball close
                    dist_error = self.target_radius - radius  # HiWonder pattern
                    # Aligned when ball is within 40 pixels of center (about 6% of frame width)
                    is_aligned = ball_x_error < 40

                    # STATE MACHINE LOGIC
                    if self.tracking_state == 'SEARCHING':
                        self.tracking_state = 'ALIGNING'
                        self.state_frames = 0

                    self.state_frames += 1

                    # State transitions
                    if self.tracking_state == 'ALIGNING':
                        if is_aligned and self.state_frames > 5:
                            self.tracking_state = 'APPROACHING'
                            self.state_frames = 0
                    elif self.tracking_state == 'APPROACHING':
                        if not is_aligned:
                            # Lost alignment, go back to aligning
                            self.tracking_state = 'ALIGNING'
                            self.state_frames = 0
                        elif abs(dist_error) <= self.radius_deadzone:
                            self.tracking_state = 'HOLDING'
                            self.state_frames = 0
                    elif self.tracking_state == 'HOLDING':
                        if abs(dist_error) > self.radius_deadzone * 2:
                            self.tracking_state = 'APPROACHING'
                            self.state_frames = 0
                        elif not is_aligned:
                            self.tracking_state = 'ALIGNING'
                            self.state_frames = 0

                    # ========== HIWONDER PATTERN: SERVO ERROR FOR ROTATION ==========
                    # From soccer_robot.py: Use servo deviation from center
                    # servo_error = servo_x - servo_center (1500)
                    # When servo turns right (servo_x > 1500), error is positive
                    # Robot should turn RIGHT to re-center camera
                    servo_error = self.servo_x - self.servo_center

                    # PID on servo error to bring camera back to center
                    self.chassis_yaw_pid.SetPoint = 0
                    self.chassis_yaw_pid.update(servo_error)

                    # EXACT HiWonder formula: NEGATE the combined PID outputs
                    # angular_z = -(chassis_yaw_pid.output + servo_x_pid.output * 0.02)
                    angular = -(self.chassis_yaw_pid.output + self.servo_x_pid.output * 0.02)
                    angular = np.clip(angular, -1.2, 1.2)

                    # ========== DISTANCE-BASED SPEED CONTROL ==========
                    # Use estimated real-world distance for smoother approach
                    # Falls back to radius-based if distance unavailable
                    if self.current_distance_cm is not None:
                        # Distance-based speed (more accurate)
                        if self.current_distance_cm > 60:
                            linear_x = 0.40  # VERY FAR - fast pursuit
                        elif self.current_distance_cm > 40:
                            linear_x = 0.30  # FAR - normal pursuit
                        elif self.current_distance_cm > 25:
                            linear_x = 0.20  # MEDIUM - controlled approach
                        elif self.current_distance_cm > 15:
                            linear_x = 0.10  # CLOSE - slow approach
                        else:
                            linear_x = 0.0   # VERY CLOSE - stop
                    else:
                        # Fallback to radius-based (original method)
                        if radius < 40:
                            linear_x = 0.35  # Ball FAR
                        elif radius < 60:
                            linear_x = 0.25  # Ball MEDIUM
                        elif radius < 85:
                            linear_x = 0.15  # Ball CLOSE
                        else:
                            linear_x = 0.05  # Ball VERY CLOSE

                    # State-specific behavior
                    if self.tracking_state == 'ALIGNING':
                        linear_x = 0.0  # No forward until aligned
                        angular = np.clip(angular * 1.5, -1.5, 1.5)  # Stronger rotation
                    elif self.tracking_state == 'HOLDING':
                        linear_x = 0.0  # Stay in place
                        angular = np.clip(angular * 0.5, -0.5, 0.5)  # Gentle corrections

                    # Send velocity command via cmd_vel (smoothing done inside)
                    self.set_velocity(linear_x, angular)

                # Log tracking info periodically
                if self.frame_count % 30 == 0:
                    self.get_logger().info(
                        f"TRACKING {target['class_name']}: "
                        f"pos=({cx},{cy}) r={radius} servo=({self.servo_x},{self.servo_y})"
                    )

            else:
                # No ball detected - but only search if truly lost (not brief gap)
                frames_since_seen = self.frame_count - self.last_seen_frame

                # If we saw ball recently (within 30 frames), keep camera steady
                if frames_since_seen < self.min_lost_before_search:
                    # Brief gap - hold position, don't search yet
                    self.lost_count = 0  # Don't accumulate lost count for brief gaps
                    return  # Skip search logic

                self.lost_count += 1
                self.last_x = None

                # Reset tracking state only after confirmed loss
                if self.tracking_state not in ['SEARCHING', 'SEARCHING_360']:
                    self.tracking_state = 'SEARCHING'
                    self.get_logger().info(f"Ball LOST for {frames_since_seen} frames - starting search")
                self.state_frames = 0

                # Stop chassis when ball is lost
                if self.chassis_enabled:
                    self.stop_chassis()
                    self.last_linear_x = 0
                    self.last_angular_z = 0

                if self.lost_count > 10 and self.lost_count < self.search_360_threshold:
                    # STAGE 1: CAMERA-ONLY SCAN (pan left/right, tilt fixed at floor)
                    self.servo_y = 2250  # 45 degrees (ideal ball tracking)

                    # Update pan position
                    self.servo_x += self.search_speed * self.scan_direction

                    # Bounce at pan limits: 800=RIGHT, 2200=LEFT
                    if self.servo_x >= 2100:  # Hit LEFT limit
                        self.scan_direction = -1  # Go RIGHT
                        self.servo_x = 2100
                    elif self.servo_x <= 900:  # Hit RIGHT limit
                        self.scan_direction = 1   # Go LEFT
                        self.servo_x = 900

                    self.last_servo_x = self.servo_x
                    self.publish_servo(self.servo_x, self.servo_y)

                elif self.lost_count == self.search_360_threshold:
                    # STAGE 2: 360 SEARCH with 90° TURNS
                    # Does 4x 90° turns with camera scan at each direction
                    # ONLY if chassis movement enabled
                    if self.chassis_enabled:
                        self.get_logger().info("Ball not found - Starting 360 SEARCH (4x 90° turns)...")
                        self.tracking_state = 'SEARCHING_360'
                        search_360(tilt=2250)
                        self.servo_x = 1500
                        self.get_logger().info("360 search complete - resuming normal scan")
                    else:
                        # Camera-only mode: just keep scanning
                        self.get_logger().info("Ball not found - camera-only mode, continuing scan")
                    self.lost_count = 0  # Reset after search

            # ==================== VISUALIZATION ====================
            # Draw ALL detections
            for det in detections:
                color = self.class_colors.get(det['class_name'], (128, 128, 128))

                # Skip drawing detections at frame edges (causes red border effect)
                if (det['x1'] <= 3 or det['y1'] <= 3 or
                    det['x2'] >= w-3 or det['y2'] >= h-3):
                    continue  # Skip edge detections

                # Clip bounding box to frame bounds
                x1 = max(3, min(det['x1'], w-3))
                y1 = max(3, min(det['y1'], h-3))
                x2 = max(3, min(det['x2'], w-3))
                y2 = max(3, min(det['y2'], h-3))

                # Draw bounding box
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

                # Draw center point (clipped)
                cx = max(5, min(det['cx'], w-5))
                cy = max(5, min(det['cy'], h-5))
                cv2.circle(result, (cx, cy), 5, color, -1)

                # Draw label (with safe positioning)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                label_y = max(20, y1 - 10)
                cv2.putText(result, label, (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw status with state machine info
            if target is not None:
                state_colors = {
                    'ALIGNING': (0, 165, 255),        # Orange
                    'APPROACHING': (0, 255, 0),       # Green
                    'HOLDING': (255, 255, 0),         # Cyan
                    'ALIGNING_CHASSIS': (255, 0, 255), # Magenta
                    'SEARCHING': (0, 255, 0),         # Green when tracking (camera-only mode)
                }
                # In camera-only mode, show TRACKING instead of state machine state
                if not self.chassis_enabled:
                    status = f"TRACKING: {target['class_name'].upper()}"
                    status_color = (0, 255, 0)  # Green
                else:
                    status = f"{self.tracking_state}: {target['class_name'].upper()}"
                    status_color = state_colors.get(self.tracking_state, (0, 255, 0))
            else:
                state_colors_search = {
                    'SEARCHING': (0, 200, 255),       # Yellow-orange
                    'SEARCHING_360': (0, 0, 255),     # Red - full search mode
                }
                status = f"{self.tracking_state}..." if self.tracking_state in state_colors_search else "SEARCHING..."
                status_color = state_colors_search.get(self.tracking_state, (0, 200, 255))

            cv2.putText(result, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Draw detection count
            ball_count = len([d for d in detections if d['class_name'] == 'ball'])
            goal_count = len([d for d in detections if d['class_name'] == 'goalpost'])
            robot_count = len([d for d in detections if d['class_name'] == 'Robot-car'])

            cv2.putText(result, f"Ball:{ball_count} Goal:{goal_count} Robot:{robot_count}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw chassis status when tracking
            if target is not None and self.chassis_enabled:
                radius = target['radius']
                servo_error = self.servo_x - self.servo_center
                dist_error = self.target_radius - radius  # Positive = far

                # Alignment status
                if abs(servo_error) < self.align_threshold:
                    align_txt, align_col = "ALIGNED", (0, 255, 0)
                else:
                    direction = "RIGHT" if servo_error < 0 else "LEFT"
                    align_txt, align_col = f"TURN {direction}", (0, 165, 255)

                # Distance status (dist_error: positive = far, negative = close)
                if abs(dist_error) <= self.radius_deadzone:
                    dist_txt, dist_col = "DIST OK", (0, 255, 0)
                elif dist_error > 0:
                    dist_txt, dist_col = "APPROACH", (255, 165, 0)  # Ball far, go forward
                else:
                    dist_txt, dist_col = "BACK UP", (0, 0, 255)  # Ball close, go backward

                cv2.putText(result, align_txt, (w-120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, align_col, 2)
                cv2.putText(result, dist_txt, (w-120, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_col, 2)
                cv2.putText(result, f"r={int(radius)} tgt={self.target_radius}",
                           (w-150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Draw crosshair - RED for visibility
            cv2.line(result, (w//2 - 25, h//2), (w//2 + 25, h//2), (0, 0, 255), 2)
            cv2.line(result, (w//2, h//2 - 25), (w//2, h//2 + 25), (0, 0, 255), 2)
            cv2.circle(result, (w//2, h//2), 3, (0, 0, 255), -1)  # Center dot

            # Draw servo info
            cv2.putText(result, f"pan={self.servo_x} tilt={self.servo_y}",
                       (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw distance info (top right)
            if self.current_distance_cm is not None:
                # Color based on zone
                zone_colors = {
                    "VERY_CLOSE": (0, 0, 255),    # Red - stop
                    "CLOSE": (0, 165, 255),       # Orange - slow
                    "MEDIUM": (0, 255, 255),      # Yellow - normal
                    "FAR": (0, 255, 0),           # Green - go
                    "VERY_FAR": (255, 0, 0),      # Blue - fast
                    "UNKNOWN": (128, 128, 128),   # Gray
                }
                dist_color = zone_colors.get(self.current_distance_zone, (255, 255, 255))
                cv2.putText(result, f"DIST: {self.current_distance_cm:.0f}cm",
                           (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
                cv2.putText(result, f"[{self.current_distance_zone}]",
                           (w-150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 2)

            # Draw legend
            y_offset = 85
            for class_name, color in self.class_colors.items():
                cv2.rectangle(result, (10, y_offset), (25, y_offset + 15), color, -1)
                cv2.putText(result, class_name, (30, y_offset + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20

            # ==================== PUBLISH DEBUG IMAGE ====================
            out = Image()
            out.header = msg.header
            out.height, out.width = h, w
            out.encoding = "bgr8"
            out.step = w * 3
            out.data = result.tobytes()
            self.debug_pub.publish(out)

        except Exception as e:
            self.get_logger().error(f"Callback error: {str(e)}")


def main():
    rclpy.init()
    node = YOLOSoccerTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_chassis()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
