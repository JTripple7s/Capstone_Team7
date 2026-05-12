# kevin_robot

Working YOLO ball-tracker for the HiWonder TurboPi (Raspberry Pi 5).

## File

- `zhui_yolo_host.py` - main tracker. Single Python process, runs locally on the Pi.

## What it does

1. Captures camera frame (320x320)
2. Runs YOLOv8n (NCNN backend) to detect ball / goalpost / opponent robot
3. Filters out false positives (low confidence, too small, robot's own wheels)
4. Computes pixel offset + estimated distance
5. State machine: SEARCH -> CHASING -> APPROACH -> CLOSE -> TOUCH -> ARRIVED
6. Sends HiWonder serial packets (0xAA 0x55 ...) to the motor board over /dev/rrc

## Run

```bash
python3 zhui_yolo_host.py
```

Make sure the V5 NCNN model directory is at the path set in `MODEL_PATH` inside the script.

## Notes

- Runs untethered on WiFi + battery
- No ROS, no offboard compute
- Hailo-8 NPU version is `zhui_yolo_hailo.py` (not in this branch)
