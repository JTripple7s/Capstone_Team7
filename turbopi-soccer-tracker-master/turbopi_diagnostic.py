#!/usr/bin/env python3
"""
TurboPi Diagnostic & Testing Script
====================================
FOR FUTURE CLAUDE CODE SESSIONS: Read this file to understand the robot's
hardware mappings and how to troubleshoot common issues.

This script was created after extensive testing to document what ACTUALLY
works vs what the documentation says.

CRITICAL KNOWLEDGE FOR CLAUDE CODE:
===================================

1. ROBOT CONNECTION:
   - Default IP: 192.168.1.126 or 192.168.1.127 (can change!)
   - SSH: pi@<IP> password: raspberry
   - If connection fails, scan network: arp -a | grep 192.168.1
   - Robot runs Docker container "TurboPi" with ROS2 Humble

   *** CRITICAL: MUST RUN COMMANDS INSIDE DOCKER CONTAINER! ***
   Running directly on Pi (outside Docker) will NOT work for motors!

   CORRECT way to run commands from your Mac:
   ssh pi@<IP> "docker exec TurboPi bash -c 'python3 your_script.py'"

   Or SSH in first, then:
   docker exec -it TurboPi bash
   python3 your_script.py

2. DIRECT SERIAL IS MORE RELIABLE THAN ROS2:
   - Serial port: /dev/rrc
   - Baud rate: 1000000
   - Protocol: 0xAA 0x55 <func> <len> <data...> <crc8>
   - Function 3 = Motors, Function 4 = Servos
   - ROS2 cmd_vel often doesn't work - use direct serial!

   MOTOR COMMAND FORMAT (CRITICAL - must use this exact format!):
   - Data: [0x05, num_motors] + packed motor data
   - Each motor: struct.pack("<Bf", motor_id - 1, float(duty))
   - motor_id is 0-INDEXED (0,1,2,3 not 1,2,3,4)
   - duty is a 4-byte FLOAT, not an integer!
   - WRONG: [motor_id, speed & 0xFF, (speed >> 8) & 0xFF]
   - CORRECT: struct.pack("<Bf", motor_id - 1, float(duty))

3. SERVO MAPPING (CONFIRMED BY PHYSICAL TESTING):
   - Servo 1 = TILT (vertical/up-down)
     * 1200 = 90 DEGREES UP (looking straight at sky)
     * 2250 = 45 DEGREES (ideal tracking angle)
     * 2500 = DOWN (looking at floor)

   - Servo 2 = PAN (horizontal/left-right)
     * 500 = RIGHT (from robot's perspective)
     * 1500 = CENTER
     * 2500 = LEFT (from robot's perspective)

   USEFUL TILT ANGLES (all tested):
     * 1200 = 90 degrees UP (looking at sky) - MIN LIMIT
     * 2200 = ~55 degrees down
     * 2250 = ~45 degrees down (BEST FOR BALL TRACKING)
     * 2300 = ~40 degrees down
     * 2500 = ~20-25 degrees down (MAX LIMIT - can't go lower)

   WARNING: HiWonder documentation and code comments often say
   Servo1=Pan, Servo2=Tilt - THIS IS WRONG! We tested it physically.

   SMOOTH CAMERA TRACKING SETTINGS (tested and working):
     * step = 50 (small steps for smooth motion)
     * delay = 0.2 seconds (consistent timing)
     * duration = 0.18 seconds (servo movement time)
     * These settings give smooth, consistent movement in both directions

4. MOTOR MAPPING (CONFIRMED BY INDIVIDUAL WHEEL TESTING):
   - Motor 1 = Front-Left wheel
     * NEGATIVE value = wheel goes FORWARD
     * POSITIVE value = wheel goes BACKWARD

   - Motor 2 = Front-Right wheel
     * POSITIVE value = wheel goes FORWARD
     * NEGATIVE value = wheel goes BACKWARD

   - Motor 3 = Rear-Left wheel
     * NEGATIVE value = wheel goes FORWARD
     * POSITIVE value = wheel goes BACKWARD

   - Motor 4 = Rear-Right wheel
     * POSITIVE value = wheel goes FORWARD
     * NEGATIVE value = wheel goes BACKWARD

   PATTERN SUMMARY:
   - Left side motors (1, 3): INVERTED (negative = forward)
   - Right side motors (2, 4): NORMAL (positive = forward)

5. MOVEMENT PATTERNS (ALL TESTED AND WORKING):

   BASIC MOVEMENTS:
   - FORWARD:  M1=-speed, M2=+speed, M3=-speed, M4=+speed
   - BACKWARD: M1=+speed, M2=-speed, M3=+speed, M4=-speed
   - STRAFE LEFT:  M1=+speed, M2=+speed, M3=-speed, M4=-speed
   - STRAFE RIGHT: M1=-speed, M2=-speed, M3=+speed, M4=+speed

   SPIN IN PLACE:
   - SPIN LEFT (CCW):  M1=+speed, M2=+speed, M3=+speed, M4=+speed (all positive!)
   - SPIN RIGHT (CW):  M1=-speed, M2=-speed, M3=-speed, M4=-speed (all negative!)

   DIAGONAL (like chess piece - body stays straight):
   - DIAGONAL FORWARD-LEFT:  M1=0, M2=+speed, M3=-speed, M4=0 (FR + RL only)
   - DIAGONAL FORWARD-RIGHT: M1=-speed, M2=0, M3=0, M4=+speed (FL + RR only)

   ARC TURNS (like a car changing lanes):
   - ARC LEFT:  M1=-20, M2=+80, M3=-20, M4=+80 (left slow, right fast)
   - ARC RIGHT: M1=-80, M2=+20, M3=-80, M4=+20 (left fast, right slow)

   NOTE: Arc turns are OPPOSITE of what you'd expect!
   To turn LEFT, make LEFT wheels SLOWER (not faster).
   To turn RIGHT, make RIGHT wheels SLOWER (not faster).

   COMPLEX MANEUVERS TESTED:
   - Parallel parking (forward, arc back-right, arc back-left, strafe right)
   - 3-point turn (arc forward-left, arc backward-right, forward)
   - Mecanum parking (forward, strafe sideways into spot)
   - Square pattern (forward, strafe right, backward, strafe left)

   BALL ALIGNMENT (camera + body alignment for 90 degrees):
   - FAR RIGHT: speed=42, time=0.089s per step
   - FAR LEFT:  speed=42, time=0.086s per step
   - Camera pan step: 50 units, servo duration: 0.15s, delay: 0.08s
   - Right needs slightly more rotation time than left
   - Pan range: 600 (far right) to 2400 (far left), center 1500

   180 DEGREE TURN (when ball is behind - search mode):
   - LEFT 180:  36 steps, speed=42, time=0.086s, delay=0.02s
   - RIGHT 180: 36 steps, speed=42, time=0.086s, delay=0.02s
   - Camera stays centered during spin
   - Use when ball not visible - robot turns around to find it

   90 DEGREE PURE SPIN (for 360 search - no camera tracking):
   - RIGHT 90: 17 steps, speed=42, time=0.094s, delay=0.02s
   - LEFT 90:  17 steps, speed=42, time=0.094s, delay=0.02s
   - Different from alignment (which has camera tracking)
   - Use for 360 search: spin 90, scan camera, repeat 4x

   360 DEGREE BALL SEARCH (full coverage):
   - Scan FRONT (camera left-right at 45 deg)
   - Turn 90 RIGHT, scan again
   - Turn 90 RIGHT (now 180), scan again
   - Turn 90 RIGHT (now 270), scan again
   - Turn 90 RIGHT to return to start
   - Covers all directions when ball not found

6. COMMON ISSUES AND FIXES:

   Issue: Motors don't respond
   Fix: Use direct serial instead of ROS2 cmd_vel
        AND make sure you're running INSIDE Docker container!
        Commands run directly on Pi (outside Docker) won't work!

   Issue: Robot spins instead of going straight
   Fix: Check motor direction signs (see patterns above)

   Issue: SSH connection timeout
   Fix: Robot IP may have changed - scan with arp -a

   Issue: Servos jitter rapidly
   Fix: Add deadzone (25+ pixels) and only send commands when position changes

   Issue: Camera looking at sky instead of floor
   Fix: Servo 1 controls tilt, set to 2500 for floor

   Issue: USB-C power causes issues
   Fix: Use battery power - USB-C doesn't provide enough current for motors

   *** CRITICAL HARDWARE NOTES ***

   POWER SUPPLY:
   - DO NOT use USB-C wall adapter for motors - not enough current!
   - USB-C causes: intermittent motor response, network drops, random behavior
   - ALWAYS use the battery pack for motor testing and ball tracking
   - USB-C is OK for just SSH/coding, but motors NEED battery power

   CAMERA MOUNT SCREW:
   - The camera mount screw CAN COME LOOSE during operation!
   - Symptoms: camera angle drifts, inconsistent pan/tilt positions
   - Fix: physically check and tighten the screw on camera mount
   - After tightening, RE-RUN camera tests to recalibrate:
     python3 turbopi_diagnostic.py --test-servos
   - This happened during our testing session - always check hardware first!

   IF THINGS STOP WORKING:
   1. Check battery is connected and charged
   2. Check camera mount screw is tight
   3. Run diagnostic tests to verify: python3 turbopi_diagnostic.py --test-all
   4. Check robot IP hasn't changed: arp -a | grep 192.168.1
   5. Restart robot if needed, wait 30 seconds for boot

7. DOCKER COMMANDS:
   Run on robot via SSH:
   - Enter container: docker exec -it TurboPi bash
   - Source ROS2: source /opt/ros/humble/setup.bash && source /home/ubuntu/ros2_ws/install/setup.bash
   - Run tracker: python3 /home/ubuntu/yolo_soccer_tracker.py

Author: Capstone Team 7 (tested with Claude Code assistance)
Robot: HiWonder TurboPi (Raspberry Pi 5 + ROS2 Humble)
"""

import serial
import struct
import time
import sys
import argparse
import subprocess

def kill_conflicting_processes():
    """Kill ball tracking scripts that might conflict."""
    conflicts = ['simple_ball', 'ball_track', 'yolo_soccer', '/app/tracking']
    for name in conflicts:
        try:
            subprocess.run(['pkill', '-f', name], capture_output=True)
        except:
            pass
    time.sleep(0.3)

# ============ CRC8 TABLE FOR SERIAL PROTOCOL ============
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


class TurboPiDiagnostic:
    """Diagnostic and testing class for TurboPi robot."""

    def __init__(self, port="/dev/rrc", baudrate=1000000):
        """Initialize serial connection to robot controller."""
        self.port = None
        self.port_name = port
        self.baudrate = baudrate

    def connect(self):
        """Open serial connection."""
        try:
            self.port = serial.Serial(self.port_name, self.baudrate, timeout=5)
            self.port.rts = False
            self.port.dtr = False
            print(f"[OK] Connected to {self.port_name} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            print("        Make sure you're running this ON the robot (via SSH)")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.port:
            self.port.close()
            print("[OK] Disconnected")

    def _crc8(self, data):
        """Calculate CRC8 checksum."""
        check = 0
        for b in data:
            check = CRC8_TABLE[check ^ b]
        return check & 0xFF

    def _send(self, func, data):
        """Send command via serial protocol."""
        if not self.port:
            print("[ERROR] Not connected")
            return False
        buf = [0xAA, 0x55, int(func), len(data)] + list(data)
        buf.append(self._crc8(bytes(buf[2:])))
        try:
            self.port.write(bytes(buf))
            return True
        except Exception as e:
            print(f"[ERROR] Send failed: {e}")
            return False

    # ============ SERVO CONTROL ============

    def servo_set(self, positions, duration=0.5):
        """
        Set servo positions.

        Args:
            positions: list of [servo_id, position] pairs
                       servo_id: 1=TILT, 2=PAN
                       position: 500-2500
            duration: movement time in seconds
        """
        duration_ms = int(duration * 1000)
        data = [0x01, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, len(positions)]
        for servo_id, pos in positions:
            data.extend(struct.pack("<BH", servo_id, pos))
        return self._send(4, data)

    def servo_tilt(self, position, duration=0.5):
        """Move tilt servo (Servo 1). 1000=UP, 2500=DOWN"""
        print(f"[TILT] Setting to {position} (1000=UP, 2500=DOWN)")
        return self.servo_set([[1, position]], duration)

    def servo_pan(self, position, duration=0.5):
        """Move pan servo (Servo 2). 500=RIGHT, 1500=CENTER, 2500=LEFT"""
        print(f"[PAN] Setting to {position} (500=RIGHT, 1500=CENTER, 2500=LEFT)")
        return self.servo_set([[2, position]], duration)

    def camera_smooth_track(self, start_pan=1500, end_pan=800, tilt=2250, step=50, delay=0.2):
        """
        Smooth camera tracking movement. TESTED AND WORKING.

        OPTIMAL SETTINGS (tested):
        - step=50 (small steps for smoothness)
        - delay=0.2 (consistent timing between steps)
        - duration=0.18 (servo movement time)
        - tilt=2250 (45 degrees - best for ball tracking)

        Pan range: 500 (far right) to 2500 (far left)
        """
        duration = delay - 0.02  # Slightly less than delay for smooth motion

        if start_pan < end_pan:
            # Moving left
            for pan in range(start_pan, end_pan + 1, step):
                self.servo_set([[1, tilt], [2, pan]], duration)
                time.sleep(delay)
        else:
            # Moving right
            for pan in range(start_pan, end_pan - 1, -step):
                self.servo_set([[1, tilt], [2, pan]], duration)
                time.sleep(delay)

    def camera_full_scan(self, tilt=2250):
        """
        Full camera scan from right to left and back. TESTED AND WORKING.
        Uses smooth tracking with optimal settings.
        """
        step = 50
        delay = 0.2
        duration = 0.18

        print("[CAMERA] Starting full scan at 45 degrees")
        self.servo_set([[1, tilt], [2, 1500]], 0.5)
        time.sleep(0.5)

        print("[CAMERA] Scanning to FAR RIGHT (500)...")
        for pan in range(1500, 500, -step):
            self.servo_set([[1, tilt], [2, pan]], duration)
            time.sleep(delay)
        time.sleep(0.5)

        print("[CAMERA] Scanning to FAR LEFT (2500)...")
        for pan in range(500, 2500, step):
            self.servo_set([[1, tilt], [2, pan]], duration)
            time.sleep(delay)
        time.sleep(0.5)

        print("[CAMERA] Return to center...")
        for pan in range(2500, 1500, -step):
            self.servo_set([[1, tilt], [2, pan]], duration)
            time.sleep(delay)

        print("[CAMERA] Scan complete!")

    # ============ MOTOR CONTROL ============

    def motor_set(self, speeds):
        """
        Set motor speeds.

        Args:
            speeds: list of [motor_id, duty] pairs
                    motor_id: 1-4
                    duty: -100 to 100
        """
        data = [0x05, len(speeds)]
        for motor_id, duty in speeds:
            data.extend(struct.pack("<Bf", motor_id - 1, float(duty)))
        return self._send(3, data)

    def motor_stop(self):
        """Stop all motors."""
        print("[MOTORS] Stopping all")
        return self.motor_set([[1, 0], [2, 0], [3, 0], [4, 0]])

    def motor_forward(self, speed=60, duration=1.0):
        """Drive forward. Left motors negative, right motors positive."""
        print(f"[MOTORS] Forward at speed {speed} for {duration}s")
        print(f"         Pattern: M1=-{speed}, M2=+{speed}, M3=-{speed}, M4=+{speed}")
        self.motor_set([[1, -speed], [2, speed], [3, -speed], [4, speed]])
        time.sleep(duration)
        self.motor_stop()

    def motor_backward(self, speed=60, duration=1.0):
        """Drive backward."""
        print(f"[MOTORS] Backward at speed {speed} for {duration}s")
        print(f"         Pattern: M1=+{speed}, M2=-{speed}, M3=+{speed}, M4=-{speed}")
        self.motor_set([[1, speed], [2, -speed], [3, speed], [4, -speed]])
        time.sleep(duration)
        self.motor_stop()

    def motor_spin_left(self, speed=60, duration=1.0):
        """Spin counter-clockwise (left). TESTED: All motors POSITIVE."""
        print(f"[MOTORS] Spin LEFT at speed {speed} for {duration}s")
        print(f"         Pattern: All motors positive")
        self.motor_set([[1, speed], [2, speed], [3, speed], [4, speed]])
        time.sleep(duration)
        self.motor_stop()

    def motor_spin_right(self, speed=60, duration=1.0):
        """Spin clockwise (right). TESTED: All motors NEGATIVE."""
        print(f"[MOTORS] Spin RIGHT at speed {speed} for {duration}s")
        print(f"         Pattern: All motors negative")
        self.motor_set([[1, -speed], [2, -speed], [3, -speed], [4, -speed]])
        time.sleep(duration)
        self.motor_stop()

    def motor_diagonal_forward_left(self, speed=80, duration=1.5):
        """Move diagonally forward-left (like chess piece). TESTED."""
        print(f"[MOTORS] Diagonal FORWARD-LEFT at speed {speed} for {duration}s")
        print(f"         Pattern: M1=0, M2=+{speed}, M3=-{speed}, M4=0 (FR + RL only)")
        self.motor_set([[1, 0], [2, speed], [3, -speed], [4, 0]])
        time.sleep(duration)
        self.motor_stop()

    def motor_diagonal_forward_right(self, speed=80, duration=1.5):
        """Move diagonally forward-right (like chess piece). TESTED."""
        print(f"[MOTORS] Diagonal FORWARD-RIGHT at speed {speed} for {duration}s")
        print(f"         Pattern: M1=-{speed}, M2=0, M3=0, M4=+{speed} (FL + RR only)")
        self.motor_set([[1, -speed], [2, 0], [3, 0], [4, speed]])
        time.sleep(duration)
        self.motor_stop()

    def motor_arc_left(self, speed=80, duration=1.5):
        """Arc left while moving forward (like car changing to left lane).
        TESTED AND WORKING: Left wheels SLOWER, right wheels FASTER.
        """
        slow = int(speed * 0.25)  # 25% speed for inside wheels
        print(f"[MOTORS] Arc LEFT at speed {speed} for {duration}s")
        print(f"         Pattern: M1=-{slow}, M2=+{speed}, M3=-{slow}, M4=+{speed}")
        print(f"         (Left wheels slow, right wheels fast)")
        self.motor_set([[1, -slow], [2, speed], [3, -slow], [4, speed]])
        time.sleep(duration)
        self.motor_stop()

    def motor_arc_right(self, speed=80, duration=1.5):
        """Arc right while moving forward (like car changing to right lane).
        TESTED AND WORKING: Left wheels FASTER, right wheels SLOWER.
        """
        slow = int(speed * 0.25)  # 25% speed for inside wheels
        print(f"[MOTORS] Arc RIGHT at speed {speed} for {duration}s")
        print(f"         Pattern: M1=-{speed}, M2=+{slow}, M3=-{speed}, M4=+{slow}")
        print(f"         (Left wheels fast, right wheels slow)")
        self.motor_set([[1, -speed], [2, slow], [3, -speed], [4, slow]])
        time.sleep(duration)
        self.motor_stop()

    # ============ COMPLEX MANEUVERS (TESTED) ============

    def maneuver_parallel_park(self, speed=60):
        """Parallel parking maneuver. TESTED AND WORKING."""
        print("\n=== PARALLEL PARKING ===")

        print("Step 1: Drive forward past spot")
        self.motor_set([[1, -speed], [2, speed], [3, -speed], [4, speed]])
        time.sleep(1)
        self.motor_stop()
        time.sleep(0.5)

        print("Step 2: Arc backward-right into spot")
        self.motor_set([[1, speed], [2, -int(speed*0.25)], [3, speed], [4, -int(speed*0.25)]])
        time.sleep(1.2)
        self.motor_stop()
        time.sleep(0.5)

        print("Step 3: Arc backward-left to straighten")
        self.motor_set([[1, int(speed*0.25)], [2, -speed], [3, int(speed*0.25)], [4, -speed]])
        time.sleep(1.2)
        self.motor_stop()
        time.sleep(0.5)

        print("Step 4: Strafe right to curb")
        self.motor_set([[1, -speed], [2, -speed], [3, speed], [4, speed]])
        time.sleep(0.8)
        self.motor_stop()
        print("PARKED!")

    def maneuver_three_point_turn(self, speed=70):
        """3-point turn (K-turn). TESTED AND WORKING."""
        print("\n=== 3-POINT TURN ===")

        print("Step 1: Arc forward-left")
        self.motor_set([[1, -int(speed*0.3)], [2, speed], [3, -int(speed*0.3)], [4, speed]])
        time.sleep(1.2)
        self.motor_stop()
        time.sleep(0.5)

        print("Step 2: Arc backward-right")
        self.motor_set([[1, speed], [2, -int(speed*0.3)], [3, speed], [4, -int(speed*0.3)]])
        time.sleep(1.2)
        self.motor_stop()
        time.sleep(0.5)

        print("Step 3: Forward to complete turn")
        self.motor_set([[1, -speed], [2, speed], [3, -speed], [4, speed]])
        time.sleep(1)
        self.motor_stop()
        print("TURNED AROUND!")

    def maneuver_mecanum_park(self, speed=70):
        """Mecanum parking - slide sideways into spot. TESTED AND WORKING."""
        print("\n=== MECANUM PARKING ===")
        print("(Only mecanum wheels can do this!)")

        print("Step 1: Drive forward past spot")
        self.motor_set([[1, -speed], [2, speed], [3, -speed], [4, speed]])
        time.sleep(1.2)
        self.motor_stop()
        time.sleep(0.5)

        print("Step 2: Slide RIGHT into parking spot")
        self.motor_set([[1, -speed], [2, -speed], [3, speed], [4, speed]])
        time.sleep(1.5)
        self.motor_stop()
        print("PARKED! (No turning needed!)")

    def maneuver_ball_track_simulation(self, speed=40):
        """
        Simulate ball tracking - car moves forward while camera scans.
        TESTED AND WORKING.

        This is what ball tracking looks like:
        - Car moves forward slowly
        - Camera scans left-right at 45 degrees
        - Smooth movement to avoid blur
        """
        print("\n=== BALL TRACKING SIMULATION ===")

        # Smooth camera settings (tested - avoids blur)
        step = 50
        delay = 0.25
        duration = 0.22
        tilt = 2250  # 45 degrees

        # Start: camera at center, 45 degrees
        self.servo_set([[1, tilt], [2, 1500]], 0.5)
        time.sleep(1)

        # Start moving forward slowly
        print("Car moving forward, camera scanning...")
        self.motor_set([[1, -speed], [2, speed], [3, -speed], [4, speed]])

        # Camera scans right while moving
        for pan in range(1500, 900, -step):
            self.servo_set([[1, tilt], [2, pan]], duration)
            time.sleep(delay)

        # Camera scans left while moving
        for pan in range(900, 2100, step):
            self.servo_set([[1, tilt], [2, pan]], duration)
            time.sleep(delay)

        # Camera back to center
        for pan in range(2100, 1500, -step):
            self.servo_set([[1, tilt], [2, pan]], duration)
            time.sleep(delay)

        # Stop car
        self.motor_stop()
        print("Done! Car stopped, camera at center")

    def maneuver_align_to_ball(self, direction="right", tilt=2250):
        """
        Align car body to where camera is looking (ball position).
        TESTED AND TUNED FOR 90 DEGREE ALIGNMENT.

        When camera sees ball on right/left, car rotates so front chassis
        faces that direction. Camera returns to center as car rotates.

        TUNED SETTINGS (tested extensively):
        - FAR RIGHT (90 deg): speed=42, time=0.089s
        - FAR LEFT (90 deg):  speed=42, time=0.086s
        - Note: Right needs slightly more time than left

        Args:
            direction: "right" or "left" - where the ball is
            tilt: camera tilt angle (default 2250 = 45 degrees)
        """
        if direction == "right":
            print("\n=== ALIGN TO BALL (on RIGHT - 90 degrees) ===")
            print("Step 1: Camera looks FAR RIGHT at ball position")
            self.servo_set([[1, tilt], [2, 600]], 0.5)
            time.sleep(0.6)

            print("Step 2: Car rotates RIGHT while camera returns to center")
            # TUNED: speed=42, time=0.089s for 90 degree right rotation
            for pan in range(600, 1550, 50):
                self.motor_set([[1, -42], [2, -42], [3, -42], [4, -42]])
                time.sleep(0.089)
                self.motor_stop()
                self.servo_set([[1, tilt], [2, pan]], 0.15)
                time.sleep(0.08)

        else:  # left
            print("\n=== ALIGN TO BALL (on LEFT - 90 degrees) ===")
            print("Step 1: Camera looks FAR LEFT at ball position")
            self.servo_set([[1, tilt], [2, 2400]], 0.5)
            time.sleep(0.6)

            print("Step 2: Car rotates LEFT while camera returns to center")
            # TUNED: speed=42, time=0.086s for 90 degree left rotation
            for pan in range(2400, 1450, -50):
                self.motor_set([[1, 42], [2, 42], [3, 42], [4, 42]])
                time.sleep(0.086)
                self.motor_stop()
                self.servo_set([[1, tilt], [2, pan]], 0.15)
                time.sleep(0.08)

        # Final center
        self.servo_set([[1, tilt], [2, 1500]], 0.3)
        time.sleep(0.4)
        self.motor_stop()
        print("Step 3: Car now faces where ball was")
        print("        Camera centered, chassis aligned!")

    def maneuver_turn_180(self, direction="left", tilt=2250):
        """
        Turn 180 degrees when ball is behind (not visible).
        TESTED AND TUNED.

        Use this in search mode when camera doesn't see ball -
        robot turns around to check behind.

        TUNED SETTINGS:
        - LEFT 180:  36 steps, speed=42, time=0.086s
        - RIGHT 180: 36 steps, speed=42, time=0.086s

        Args:
            direction: "left" or "right" - which way to spin
            tilt: camera tilt angle (default 2250 = 45 degrees)
        """
        print(f"\n=== TURN 180 {direction.upper()} (ball behind) ===")

        # Keep camera centered at 45 degrees
        self.servo_set([[1, tilt], [2, 1500]], 0.3)
        time.sleep(0.3)

        print(f"Spinning {direction.upper()} 180 degrees...")

        if direction == "left":
            # All motors positive for left spin
            for i in range(36):
                self.motor_set([[1, 42], [2, 42], [3, 42], [4, 42]])
                time.sleep(0.086)
                self.motor_stop()
                time.sleep(0.02)
        else:  # right
            # All motors negative for right spin
            for i in range(36):
                self.motor_set([[1, -42], [2, -42], [3, -42], [4, -42]])
                time.sleep(0.086)
                self.motor_stop()
                time.sleep(0.02)

        self.motor_stop()
        print("DONE - Now facing opposite direction!")

    def spin_90(self, direction="right"):
        """
        Pure 90 degree spin (no camera tracking).
        TUNED for 360 search mode.

        TUNED SETTINGS:
        - 17 steps, speed=42, time=0.094s, delay=0.02s

        Args:
            direction: "right" or "left"
        """
        print(f"  Spinning 90 {direction.upper()}...")
        for i in range(17):
            if direction == "right":
                self.motor_set([[1, -42], [2, -42], [3, -42], [4, -42]])
            else:
                self.motor_set([[1, 42], [2, 42], [3, 42], [4, 42]])
            time.sleep(0.094)
            self.motor_stop()
            time.sleep(0.02)
        self.motor_stop()

    def scan_camera(self, tilt=2250):
        """
        Scan camera left to right at 45 degrees.
        Used in 360 search mode.
        """
        print("  Scanning camera left-right...")
        self.servo_set([[1, tilt], [2, 1500]], 0.3)
        time.sleep(0.3)
        # Scan right
        for pan in range(1500, 600, -100):
            self.servo_set([[1, tilt], [2, pan]], 0.15)
            time.sleep(0.15)
        # Scan left
        for pan in range(600, 2400, 100):
            self.servo_set([[1, tilt], [2, pan]], 0.15)
            time.sleep(0.15)
        # Back to center
        self.servo_set([[1, tilt], [2, 1500]], 0.3)
        time.sleep(0.3)

    def maneuver_360_search(self, tilt=2250):
        """
        Full 360 degree ball search.
        TESTED AND TUNED.

        Scans in 4 directions (front, right, back, left)
        then returns to start position.

        TUNED SETTINGS:
        - 90 degree spin: 17 steps, speed=42, time=0.094s
        - Camera scan: pan 600-2400 at tilt 2250 (45 deg)

        Use when ball is not visible - covers all directions.
        """
        print("\n=== 360 DEGREE BALL SEARCH ===")
        print("Scanning in 4 directions\n")

        # Direction 1: FRONT (0 degrees)
        print("[1/4] Checking FRONT (0 deg)...")
        self.scan_camera(tilt)
        time.sleep(0.3)

        # Direction 2: RIGHT (90 degrees)
        print("[2/4] Checking RIGHT (90 deg)...")
        self.spin_90("right")
        time.sleep(0.2)
        self.scan_camera(tilt)
        time.sleep(0.3)

        # Direction 3: BACK (180 degrees)
        print("[3/4] Checking BACK (180 deg)...")
        self.spin_90("right")
        time.sleep(0.2)
        self.scan_camera(tilt)
        time.sleep(0.3)

        # Direction 4: LEFT (270 degrees)
        print("[4/4] Checking LEFT (270 deg)...")
        self.spin_90("right")
        time.sleep(0.2)
        self.scan_camera(tilt)
        time.sleep(0.3)

        # Return to original direction
        print("\nReturning to start position...")
        self.spin_90("right")

        self.motor_stop()
        self.servo_set([[1, tilt], [2, 1500]], 0.3)
        print("\n=== 360 SEARCH COMPLETE ===")

    def maneuver_square(self, speed=70):
        """Drive in a square while always facing forward. TESTED AND WORKING."""
        print("\n=== SQUARE PATTERN ===")
        print("(Robot always faces forward!)")

        print("Side 1: FORWARD")
        self.motor_set([[1, -speed], [2, speed], [3, -speed], [4, speed]])
        time.sleep(1)
        self.motor_stop()
        time.sleep(0.3)

        print("Side 2: STRAFE RIGHT")
        self.motor_set([[1, -speed], [2, -speed], [3, speed], [4, speed]])
        time.sleep(1)
        self.motor_stop()
        time.sleep(0.3)

        print("Side 3: BACKWARD")
        self.motor_set([[1, speed], [2, -speed], [3, speed], [4, -speed]])
        time.sleep(1)
        self.motor_stop()
        time.sleep(0.3)

        print("Side 4: STRAFE LEFT")
        self.motor_set([[1, speed], [2, speed], [3, -speed], [4, -speed]])
        time.sleep(1)
        self.motor_stop()
        print("BACK TO START!")

    # ============ INDIVIDUAL WHEEL TESTS ============

    def test_motor_individual(self, motor_id, speed=60, duration=1.0):
        """
        Test a single motor.

        Use this to verify motor mapping:
        - Motor 1 = Front-Left
        - Motor 2 = Front-Right
        - Motor 3 = Rear-Left
        - Motor 4 = Rear-Right
        """
        motor_names = {1: "Front-Left", 2: "Front-Right", 3: "Rear-Left", 4: "Rear-Right"}
        name = motor_names.get(motor_id, f"Motor {motor_id}")

        print(f"\n[TEST] Motor {motor_id} ({name})")
        print(f"       Positive value (+{speed}): ", end="")
        if motor_id in [1, 3]:  # Left side
            print("wheel should go BACKWARD")
        else:  # Right side
            print("wheel should go FORWARD")

        self.motor_set([[motor_id, speed]])
        time.sleep(duration)
        self.motor_stop()
        time.sleep(0.5)

        print(f"       Negative value (-{speed}): ", end="")
        if motor_id in [1, 3]:  # Left side
            print("wheel should go FORWARD")
        else:  # Right side
            print("wheel should go BACKWARD")

        self.motor_set([[motor_id, -speed]])
        time.sleep(duration)
        self.motor_stop()

    # ============ DIAGNOSTIC TEST SEQUENCES ============

    def test_all_servos(self):
        """Run complete servo test sequence."""
        print("\n" + "="*50)
        print("SERVO TEST SEQUENCE")
        print("="*50)

        print("\n[1/6] Center position (tilt=1750, pan=1500)")
        self.servo_set([[1, 1750], [2, 1500]], 1.0)
        time.sleep(1.5)

        print("\n[2/6] Tilt UP (servo 1 = 1000)")
        self.servo_tilt(1000)
        time.sleep(1.5)

        print("\n[3/6] Tilt DOWN (servo 1 = 2500)")
        self.servo_tilt(2500)
        time.sleep(1.5)

        print("\n[4/6] Pan RIGHT (servo 2 = 500)")
        self.servo_pan(500)
        time.sleep(1.5)

        print("\n[5/6] Pan LEFT (servo 2 = 2500)")
        self.servo_pan(2500)
        time.sleep(1.5)

        print("\n[6/6] Return to center")
        self.servo_set([[1, 1750], [2, 1500]], 1.0)
        time.sleep(1.0)

        print("\n[OK] Servo test complete!")

    def test_all_motors(self):
        """Run complete motor test sequence."""
        print("\n" + "="*50)
        print("MOTOR TEST SEQUENCE")
        print("="*50)
        print("Place robot on a surface where it can move freely!")
        print("Starting in 3 seconds...")
        time.sleep(3)

        print("\n[1/6] Testing Motor 1 (Front-Left)")
        self.test_motor_individual(1, 60, 0.5)
        time.sleep(1)

        print("\n[2/6] Testing Motor 2 (Front-Right)")
        self.test_motor_individual(2, 60, 0.5)
        time.sleep(1)

        print("\n[3/6] Testing Motor 3 (Rear-Left)")
        self.test_motor_individual(3, 60, 0.5)
        time.sleep(1)

        print("\n[4/6] Testing Motor 4 (Rear-Right)")
        self.test_motor_individual(4, 60, 0.5)
        time.sleep(1)

        print("\n[5/6] Testing FORWARD motion")
        self.motor_forward(60, 1.0)
        time.sleep(1)

        print("\n[6/6] Testing BACKWARD motion")
        self.motor_backward(60, 1.0)

        print("\n[OK] Motor test complete!")

    def test_full_diagnostic(self):
        """Run complete diagnostic of all systems."""
        print("\n" + "="*60)
        print("TURBOPI FULL DIAGNOSTIC TEST")
        print("="*60)
        print("""
This will test:
1. Serial connection
2. All servo movements (camera pan/tilt)
3. All motor movements (individual + combined)

Make sure robot is on a flat surface with room to move!
        """)

        input("Press ENTER to start (or Ctrl+C to cancel)...")

        self.test_all_servos()
        print("\n" + "-"*40)
        time.sleep(2)
        self.test_all_motors()

        print("\n" + "="*60)
        print("DIAGNOSTIC COMPLETE")
        print("="*60)


def print_reference():
    """Print quick reference card."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           TURBOPI QUICK REFERENCE CARD                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  SERVO MAPPING (Function 4):                                 ║
║  ┌─────────┬──────────┬─────────────────────────────┐        ║
║  │ Servo   │ Function │ Values                      │        ║
║  ├─────────┼──────────┼─────────────────────────────┤        ║
║  │ Servo 1 │ TILT     │ 1000=UP, 2500=DOWN          │        ║
║  │ Servo 2 │ PAN      │ 500=RIGHT, 1500=CTR, 2500=L │        ║
║  └─────────┴──────────┴─────────────────────────────┘        ║
║                                                              ║
║  MOTOR MAPPING (Function 3):                                 ║
║  ┌─────────┬─────────────┬──────────────────────────┐        ║
║  │ Motor   │ Position    │ Forward = ?              │        ║
║  ├─────────┼─────────────┼──────────────────────────┤        ║
║  │ Motor 1 │ Front-Left  │ NEGATIVE value           │        ║
║  │ Motor 2 │ Front-Right │ POSITIVE value           │        ║
║  │ Motor 3 │ Rear-Left   │ NEGATIVE value           │        ║
║  │ Motor 4 │ Rear-Right  │ POSITIVE value           │        ║
║  └─────────┴─────────────┴──────────────────────────┘        ║
║                                                              ║
║  MOVEMENT PATTERNS (speed=80 example):                       ║
║  ┌───────────┬────────────────────────────────────┐          ║
║  │ Direction │ M1    M2    M3    M4               │          ║
║  ├───────────┼────────────────────────────────────┤          ║
║  │ FORWARD   │ -80   +80   -80   +80              │          ║
║  │ ARC LEFT  │ -20   +80   -20   +80 (L slow)     │          ║
║  │ ARC RIGHT │ -80   +20   -80   +20 (R slow)     │          ║
║  │ BACKWARD  │ +80   -80   +80   -80              │          ║
║  │ SPIN LEFT │ -80   -80   -80   -80              │          ║
║  │ SPIN RIGHT│ +80   +80   +80   +80              │          ║
║  └───────────┴────────────────────────────────────┘          ║
║                                                              ║
║  SERIAL PROTOCOL:                                            ║
║  Port: /dev/rrc  Baud: 1000000                               ║
║  Format: 0xAA 0x55 <func> <len> <data...> <crc8>             ║
║  Functions: 3=Motors, 4=Servos                               ║
║                                                              ║
║  COMMON ISSUES:                                              ║
║  • Motors don't work → Use direct serial, not ROS2 cmd_vel   ║
║  • Wrong IP → Scan with: arp -a | grep 192.168.1             ║
║  • Camera at sky → Set Servo 1 (TILT) to 2500 for floor      ║
║  • Robot spins → Check motor signs (left=negative)           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description="TurboPi Diagnostic & Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 turbopi_diagnostic.py --test-servos     # Test camera pan/tilt
  python3 turbopi_diagnostic.py --test-motors     # Test all wheels
  python3 turbopi_diagnostic.py --test-all        # Full diagnostic
  python3 turbopi_diagnostic.py --reference       # Print reference card
  python3 turbopi_diagnostic.py --motor 1         # Test single motor
  python3 turbopi_diagnostic.py --forward         # Drive forward 1 second
        """
    )

    parser.add_argument("--test-servos", action="store_true", help="Test all servo movements")
    parser.add_argument("--test-motors", action="store_true", help="Test all motor movements")
    parser.add_argument("--test-all", action="store_true", help="Run full diagnostic")
    parser.add_argument("--reference", action="store_true", help="Print reference card")
    parser.add_argument("--motor", type=int, choices=[1,2,3,4], help="Test individual motor")
    parser.add_argument("--forward", action="store_true", help="Drive forward")
    parser.add_argument("--backward", action="store_true", help="Drive backward")
    parser.add_argument("--spin-left", action="store_true", help="Spin left")
    parser.add_argument("--spin-right", action="store_true", help="Spin right")
    parser.add_argument("--arc-left", action="store_true", help="Arc left (car-like turn)")
    parser.add_argument("--arc-right", action="store_true", help="Arc right (car-like turn)")
    parser.add_argument("--align-right", action="store_true", help="Align car to ball on right")
    parser.add_argument("--align-left", action="store_true", help="Align car to ball on left")
    parser.add_argument("--turn-180-left", action="store_true", help="Turn 180 degrees left (ball behind)")
    parser.add_argument("--turn-180-right", action="store_true", help="Turn 180 degrees right (ball behind)")
    parser.add_argument("--spin-90-left", action="store_true", help="Pure 90 degree spin left (for search)")
    parser.add_argument("--spin-90-right", action="store_true", help="Pure 90 degree spin right (for search)")
    parser.add_argument("--search-360", action="store_true", help="Full 360 degree ball search")
    parser.add_argument("--tilt", type=int, help="Set tilt position (1000-2500)")
    parser.add_argument("--pan", type=int, help="Set pan position (500-2500)")
    parser.add_argument("--speed", type=int, default=60, help="Motor speed (default: 60)")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration in seconds (default: 1.0)")
    parser.add_argument("--port", default="/dev/rrc", help="Serial port (default: /dev/rrc)")

    args = parser.parse_args()

    # Just print reference if requested
    if args.reference:
        print_reference()
        return

    # Kill any conflicting processes before taking control
    print("[INIT] Killing conflicting processes...")
    kill_conflicting_processes()

    # If no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*50)
        print("TIP: Run with --reference for quick reference card")
        print("="*50)
        return

    # Create diagnostic instance and connect
    diag = TurboPiDiagnostic(args.port)
    if not diag.connect():
        sys.exit(1)

    try:
        if args.test_all:
            diag.test_full_diagnostic()
        elif args.test_servos:
            diag.test_all_servos()
        elif args.test_motors:
            diag.test_all_motors()
        elif args.motor:
            diag.test_motor_individual(args.motor, args.speed, args.duration)
        elif args.forward:
            diag.motor_forward(args.speed, args.duration)
        elif args.backward:
            diag.motor_backward(args.speed, args.duration)
        elif args.spin_left:
            diag.motor_spin_left(args.speed, args.duration)
        elif args.spin_right:
            diag.motor_spin_right(args.speed, args.duration)
        elif args.arc_left:
            diag.motor_arc_left(args.speed, args.duration)
        elif args.arc_right:
            diag.motor_arc_right(args.speed, args.duration)
        elif args.align_right:
            diag.maneuver_align_to_ball("right")
        elif args.align_left:
            diag.maneuver_align_to_ball("left")
        elif args.turn_180_left:
            diag.maneuver_turn_180("left")
        elif args.turn_180_right:
            diag.maneuver_turn_180("right")
        elif args.spin_90_left:
            diag.spin_90("left")
        elif args.spin_90_right:
            diag.spin_90("right")
        elif args.search_360:
            diag.maneuver_360_search()
        elif args.tilt is not None:
            diag.servo_tilt(args.tilt)
        elif args.pan is not None:
            diag.servo_pan(args.pan)
    finally:
        diag.motor_stop()
        diag.disconnect()


if __name__ == "__main__":
    main()
