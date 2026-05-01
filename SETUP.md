# Setup Guide — zhui_yolo_host.py

Step-by-step instructions to get the YOLO ball tracker running on your computer or robot.

---

## What This Is

`zhui_yolo_host.py` is a real-time YOLO-based ball tracker written for the HiWonder TurboPi (Raspberry Pi 5).
It can run in two modes:

- **Full mode** — on the actual TurboPi robot with motors, camera, and HiWonder serial board (Linux only).
- **Detection-only mode** — on any PC (Windows/Mac/Linux) with a webcam. Motors won't move (no HiWonder board), but YOLO detection + bounding boxes + state machine output will still work.

---

## Requirements

### Hardware
| Mode | Needs |
|------|-------|
| Full robot | HiWonder TurboPi + Pi 5 + camera + battery + motor board |
| PC detection-only | Any laptop with a webcam |

### Software
- **Python 3.10 or newer**
- A webcam (built-in laptop cam works fine)
- ~2 GB free disk space (for model + dependencies)

---

## Step 1 — Install Python

### Windows
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (latest stable)
3. Run the installer — **CHECK the box that says "Add Python to PATH"**
4. Open Command Prompt and verify:
   ```cmd
   python --version
   ```

### Mac
```bash
brew install python@3.11
```

### Linux / Raspberry Pi
Already installed on Pi OS.

---

## Step 2 — Clone the Repo

Open Command Prompt (Windows) or Terminal (Mac/Linux):

```bash
git clone https://github.com/JTripple7s/Capstone_Team7.git
cd Capstone_Team7
git checkout kevin_robot
```

---

## Step 3 — Install Python Packages

### Windows
```cmd
pip install opencv-python numpy ultralytics pyserial ncnn
```

### Mac / Linux
```bash
pip3 install opencv-python numpy ultralytics pyserial ncnn
```

If you get a "permission denied" error on Linux/Pi, add `--break-system-packages`:
```bash
pip3 install --break-system-packages opencv-python numpy ultralytics pyserial ncnn
```

---

## Step 4 — Download the YOLO Model

The script expects an NCNN model directory. Two options:

### Option A — Use the V5 model (recommended)
Ask Kevin for the `V5_ncnn_model` folder. Place it in your repo folder so it looks like:
```
Capstone_Team7/
├── zhui_yolo_host.py
└── V5_ncnn_model/
    ├── model.ncnn.bin
    └── model.ncnn.param
```

### Option B — Use any YOLOv8 model
Edit `zhui_yolo_host.py` and change `MODEL_PATH` to point to a `.pt` file:
```python
MODEL_PATH = 'yolov8n.pt'    # ultralytics will auto-download
```

---

## Step 5 — Configure for Your Platform

Open `zhui_yolo_host.py` in any text editor and change these lines near the top:

### Windows
```python
MODEL_PATH = './V5_ncnn_model'         # or wherever you saved it
CAMERA_DEV = 0                          # 0 = default webcam
```

### Mac
```python
MODEL_PATH = './V5_ncnn_model'
CAMERA_DEV = 0
```

### Raspberry Pi (full robot)
Already configured. Leave defaults:
```python
MODEL_PATH = '/home/pi/V5_ncnn_model'
CAMERA_DEV = '/dev/video0'
```

### PC mode — disable motor commands
Find this section and comment it out so it doesn't crash trying to find `/dev/rrc`:
```python
# SERIAL_PORT = serial.Serial("/dev/rrc", baudrate=1000000, timeout=5)
SERIAL_PORT = None   # <-- add this line
```

---

## Step 6 — Run It

### Windows
```cmd
python zhui_yolo_host.py
```

### Mac / Linux / Pi
```bash
python3 zhui_yolo_host.py
```

You should see:
```
[init] loading model: ./V5_ncnn_model
[mjpeg] http://192.168.x.x:8888
[ok] camera open
```

Open a browser and visit:
```
http://localhost:8888
```

You'll see the live camera feed with YOLO bounding boxes drawn on detected balls and goalposts.

---

## Step 7 — Stop It

Press **Ctrl+C** in the terminal.

---

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` again |
| `cannot open /dev/video0` | Change `CAMERA_DEV` to `0` for default webcam |
| `Permission denied: /dev/rrc` | Only on Pi — run with `sudo` or add user to `dialout` group |
| `port 8888 already in use` | Another instance running — kill it: `pkill -f zhui_yolo_host` |
| Model loads but no detections | Lower `CONF_THRESH` from `0.55` to `0.35` near the top of the script |
| Webcam shows but tracker freezes | Reduce `CAM_W`/`CAM_H` to `320, 240` |

---

## Notes for Windows Users

- Motors will NOT work without the HiWonder serial board (it's Linux-only `/dev/rrc`)
- The MJPEG server still works — useful for showing the model running in a browser
- If you want to test motor packets, plug in an Arduino emulating the HiWonder protocol on a COM port and change `serial.Serial("/dev/rrc", ...)` to `serial.Serial("COM3", ...)`

---

## Tested On

- Raspberry Pi 5, Pi OS Bookworm, Python 3.11 (full robot)
- macOS Sonoma, Python 3.11 (detection-only)
- Windows 11, Python 3.11 (detection-only)
