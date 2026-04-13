# TurboPi Soccer Ball Tracker

YOLO-based ball tracking system for HiWonder TurboPi robot with ROS2 Humble.

## Overview

This project implements precise ball tracking using:
- **YOLOv8** for ball detection (custom trained model `best.pt`)
- **PID control** for camera pan/tilt servo tracking
- **Direct serial communication** for motor/servo control (bypasses ROS2 latency)
- **Manually tuned motor movements** for reliable chassis control

## Hardware

- HiWonder TurboPi with Raspberry Pi 5
- 4-wheel Mecanum drive
- Pan/tilt camera servo system
- Docker container "TurboPi" for ROS2

## Key Files

### `yolo_soccer_tracker.py`
Main ball tracking node with:
- YOLO detection for ball, goalpost, Robot-car
- 2-axis camera tracking (pan + tilt) with tuned PID
- Camera-only mode for testing (set `CAMERA_ONLY_MODE = True/False`)
- Direct serial motor control (tested and working)
- All movement functions tuned through physical testing

### Serial Communication
```
Port: /dev/rrc
Baudrate: 1000000
Protocol: 0xAA 0x55 <func> <len> <data...> <crc8>
```

### Servo Mapping (Tested)
- **Servo 1 = TILT**: 1000=UP (sky), 2500=DOWN (floor), 2250=45 degrees
- **Servo 2 = PAN**: 500=RIGHT, 1500=CENTER, 2500=LEFT

### Motor Mapping (Tested individually)
- Motor 1 = Front-Left: NEGATIVE = forward
- Motor 2 = Front-Right: POSITIVE = forward
- Motor 3 = Rear-Left: NEGATIVE = forward
- Motor 4 = Rear-Right: POSITIVE = forward

## Tuned Values

### Camera PID (Stable, no oscillation)
```python
servo_x_pid = PID(P=0.20, I=0.008, D=0.008)   # Pan
servo_y_pid = PID(P=0.40, I=0.015, D=0.015)  # Tilt
deadzone_x = 8   # Horizontal
deadzone_y = 5   # Vertical
```

### Movement Timings (Manually calibrated)
```python
# 90 degree alignment
align_right: speed=42, time=0.089s, 19 steps
align_left: speed=42, time=0.086s, 19 steps

# 180 degree turn
turn_180: speed=42, time=0.086s, 36 steps

# 90 degree pure spin (for 360 search)
spin_90: speed=42, time=0.094s, 17 steps
```

## Usage

### On Robot (inside Docker)
```bash
docker exec -it TurboPi bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash
cd /home/ubuntu/ros2_ws/src/app/app
python3 yolo_soccer_tracker.py
```

### View Camera Stream
```
http://192.168.1.126:8080/stream?topic=/yolo_debug
```

### Copy files to robot
```bash
sshpass -p "raspberry" scp yolo_soccer_tracker.py pi@192.168.1.126:/tmp/
sshpass -p "raspberry" ssh pi@192.168.1.126 "docker cp /tmp/yolo_soccer_tracker.py TurboPi:/home/ubuntu/ros2_ws/src/app/app/"
```

## Development Notes

### What Works
- Camera tracks ball precisely (red crosshair on ball center)
- Pan and tilt both work with stable PID (no oscillation)
- Direct serial bypasses ROS2 latency issues
- All motor/servo mappings tested individually

### Next Steps
- Enable chassis movement (set `CAMERA_ONLY_MODE = False`)
- Implement distance estimation using ball radius in pixels
- Test full ball-following behavior

### Distance Estimation (TODO)
Use pinhole camera model:
```python
distance_cm = (ball_diameter_cm * focal_length_pixels) / ball_diameter_pixels
```
Requires one-time calibration to determine `focal_length_pixels`.

## Author
Capstone Team 7

## Technical Details

### Important: Kill Conflicting Processes
The tracker auto-kills `simple_ball`, `ball_track`, and `/app/tracking` to prevent servo conflicts.

### Tilt Range
Expanded to 1800-2500 for better vertical centering (was 2000-2500).

### PID Output Clamping
Max 30 servo units per frame to prevent over-correction.
