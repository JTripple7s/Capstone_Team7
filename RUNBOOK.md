# Operations Runbook — kevin_robot

Complete commands for running, debugging, and maintaining the bot. This covers SSH, file transfer, launching, killing, viewing live feed, and troubleshooting.

---

## Network Info

| What | Value |
|------|-------|
| Pi IP (WiFi `yomama`) | `192.168.1.126` |
| Pi IP (Ethernet direct) | `192.168.2.2` |
| Pi user | `pi` |
| Pi password | (ask Kevin) |
| MJPEG live feed | `http://<pi-ip>:8888` |
| SSH default port | `22` |

---

## SSH — Connect to the Robot

### From Mac/Linux
```bash
ssh pi@192.168.1.126
```
Enter the password when prompted.

### From Windows (PowerShell or Command Prompt)
```cmd
ssh pi@192.168.1.126
```

### Skip password prompt (one-time setup, optional)
```bash
ssh-copy-id pi@192.168.1.126
```
After this, `ssh pi@192.168.1.126` logs in without asking for a password.

### Quick one-shot command (no full login)
```bash
ssh pi@192.168.1.126 "ls /home/pi/"
ssh pi@192.168.1.126 "cat /home/pi/zhui_hailo.log | tail -30"
```

---

## File Transfer — Push Code from PC to Pi

### scp (from your laptop)
```bash
# Push tracker
scp /Users/kevinperez/Downloads/zhui_yolo_host.py pi@192.168.1.126:/home/pi/

# Push hailo version
scp /Users/kevinperez/Downloads/zhui_yolo_hailo.py pi@192.168.1.126:/home/pi/

# Push the whole NCNN model folder
scp -r /Users/kevinperez/Downloads/V5_ncnn_model pi@192.168.1.126:/home/pi/
```

### Pull files FROM the Pi
```bash
scp pi@192.168.1.126:/home/pi/zhui_hailo.log /tmp/
```

---

## Launch the Tracker

### Quick launch (foreground, see output)
```bash
ssh pi@192.168.1.126
cd /home/pi
python3 zhui_yolo_host.py
```
Press `Ctrl+C` to stop.

### Background launch (survives SSH disconnect)
SSH in, then:
```bash
# Kill anything running first
pkill -9 -f zhui_yolo_host
pkill -9 -f zhui_yolo_hailo
fuser -k 8888/tcp 2>/dev/null
sleep 2

# Clear old log
> /home/pi/zhui_host.log

# Launch detached
setsid /usr/bin/python3 /home/pi/zhui_yolo_host.py </dev/null >/home/pi/zhui_host.log 2>&1 &
echo "launched pid=$!"
```

### Launch script (`/tmp/launch_hailo.sh`)
For the Hailo NPU version, save this as `launch_hailo.sh` and run with `bash launch_hailo.sh`:
```bash
#!/bin/bash
for p in zhui_yolo_host zhui_yolo_hailo autonomous_striker bestv2_tracker hailo_tracker; do
  pkill -9 -f "$p" 2>/dev/null
done
fuser -k 8888/tcp 2>/dev/null
sleep 2
> /home/pi/zhui_hailo.log
setsid /usr/bin/python3 /home/pi/zhui_yolo_hailo.py </dev/null >/home/pi/zhui_hailo.log 2>&1 &
echo "launched pid=$!"
```

Run it remotely without logging in:
```bash
ssh pi@192.168.1.126 "bash /tmp/launch_hailo.sh"
```

---

## Kill / Stop the Tracker

```bash
ssh pi@192.168.1.126 "pkill -9 -f zhui_yolo_host"
ssh pi@192.168.1.126 "pkill -9 -f zhui_yolo_hailo"
ssh pi@192.168.1.126 "fuser -k 8888/tcp"
```

---

## View Live Camera Feed

Once the tracker is running, open any browser:
```
http://192.168.1.126:8888
```
You'll see the camera with YOLO bounding boxes drawn on detected objects.

---

## Watch Logs in Real Time

```bash
# Follow latest tracker log
ssh pi@192.168.1.126 "tail -f /home/pi/zhui_host.log"

# Last 50 lines (one shot)
ssh pi@192.168.1.126 "tail -50 /home/pi/zhui_host.log"

# Search for errors
ssh pi@192.168.1.126 "grep -iE 'error|fail|crash' /home/pi/zhui_host.log"
```

Press `Ctrl+C` to stop tailing.

---

## First-Time Pi Setup (only run once)

If setting up a fresh Pi:

### Install Python packages
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-serial python3-numpy
pip3 install --break-system-packages ultralytics ncnn
```

### Pin numpy below 2.0 (required for Hailo SDK)
```bash
pip3 install --break-system-packages 'numpy<2.0'
```

### Fix the /dev/rrc UART (motors won't work without this)
```bash
sudo nano /etc/udev/rules.d/99-serial.rules
```
Make sure ONLY `ttyAMA0` has the `rrc` symlink rule. Delete any line that maps `ttyS0` to `rrc`. Then:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -la /dev/rrc    # should point to ttyAMA0
```

### Add user to dialout group (for serial access)
```bash
sudo usermod -a -G dialout pi
# Log out and back in for group change to take effect
```

### Disable unused services (saves battery)
```bash
sudo systemctl mask hciuart        # bluetooth UART
sudo systemctl disable bluetooth
```

### Disconnect optional sensors (already done — skip if working)
- Unplug sonar from I²C (address 0x77)
- Unplug line-follower from I²C (address 0x78)

---

## Restart the Pi

```bash
ssh pi@192.168.1.126 "sudo reboot"
```
Wait ~30 seconds, then SSH back in.

---

## Power Off the Pi

```bash
ssh pi@192.168.1.126 "sudo shutdown now"
```
Wait for the green LED to stop blinking, then flip the switch.

---

## Common Operations

### Change which YOLO model is used
SSH in, then:
```bash
nano /home/pi/zhui_yolo_host.py
```
Find this line near the top and change it:
```python
MODEL_PATH = '/home/pi/V5_ncnn_model'
```
Save (Ctrl+O, Enter, Ctrl+X), then restart the tracker.

### Adjust speed (motors too fast/slow)
Find the `drive()` function in the script and change:
```python
FWD_DUTY  = 34.0   # forward speed (0-100)
TURN_DUTY = 18.0   # turn speed
MAX_ANY   = 36.0   # cap on any one motor
```

### Adjust detection threshold (false positives)
Near the top of the script:
```python
CONF_THRESH      = 0.55   # raise to reject weak detections
BALL_MIN_RADIUS  = 20     # minimum ball size in pixels
```

---

## Troubleshooting

### Bot won't move (motors silent)
1. Check `/dev/rrc` exists and points to ttyAMA0:
   ```bash
   ls -la /dev/rrc
   ```
2. Check the motor board has power (red LED should be on)
3. Make sure user is in dialout group:
   ```bash
   groups pi   # should include "dialout"
   ```
4. Test motors directly (bypass YOLO):
   ```bash
   python3 -c "import serial; s=serial.Serial('/dev/rrc', 1000000); s.write(bytes([0xAA, 0x55, 0x09, 0x06, 1, 50, 2, 50, 3, 50, 4, 50, 0]))"
   ```

### Camera not opening
```bash
ls /dev/video*       # should show /dev/video0
v4l2-ctl --list-devices
```
If missing, reseat the camera ribbon cable.

### "Port 8888 already in use"
```bash
fuser -k 8888/tcp
```

### "Cannot import name X from numpy"
Hailo SDK breaks with numpy 2.x. Downgrade:
```bash
pip3 install --break-system-packages 'numpy<2.0'
```

### MJPEG feed won't load in browser
- Confirm tracker is running: `pgrep -f zhui_yolo_host`
- Confirm port is open: `ss -tlnp | grep 8888`
- Use the right IP — `192.168.1.126` on WiFi, `192.168.2.2` on ethernet

### Brown-outs (Pi resets when motors run)
- Check battery voltage — must be above 7.4 V under load
- Use the BAK N18650COP cells (30 A continuous), not stock cells
- Inrush ramping is already enabled in the code (6 steps × 12 ms)

---

## Quick Reference (Cheat Sheet)

```bash
# Connect
ssh pi@192.168.1.126

# Push file
scp file.py pi@192.168.1.126:/home/pi/

# Launch
ssh pi@192.168.1.126 "bash /tmp/launch_hailo.sh"

# Kill
ssh pi@192.168.1.126 "pkill -9 -f zhui_yolo"

# Watch log
ssh pi@192.168.1.126 "tail -f /home/pi/zhui_hailo.log"

# View feed
open http://192.168.1.126:8888

# Reboot
ssh pi@192.168.1.126 "sudo reboot"
```

---

## Tested On

- Raspberry Pi 5, Pi OS Bookworm, Python 3.11
- HiWonder TurboPi chassis with Hailo-8 NPU (M.2 HAT)
- WiFi: `yomama` network, static IP setup recommended
- BAK N18650COP × 2 batteries (8.0 V fully charged)
