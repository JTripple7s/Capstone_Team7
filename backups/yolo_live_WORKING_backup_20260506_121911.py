#!/usr/bin/env python3
"""
yolo_live.py — V7 YOLO + LOCKED CAMERA tracker.

Camera is FIXED (pan=1500, tilt=1800). Body rotates to keep ball centered in
frame, then drives forward. Camera never moves during operation. This eliminates
camera-vs-body misalignment and bang-bang oscillation.

State machine:
  WAIT (no ball) -> stop motors
  WARMUP -> ball just appeared, building persistence
  CHASE_ROT -> ball off center, pulse-rotate body to bring ball to center
  CHASE_FWD -> ball centered, drive forward (speed scales with distance)
  HOLD -> ball in scoop, no goal seen
  AIM -> ball in scoop + goal seen, pulse-rotate to align goal behind ball
  AT_GOAL -> aimed, but goal bbox is wide (bot is close to goal). Stop, ball already there.
  SCORE -> aimed, goal not too close, full forward push (SCORE_DURATION_S)

Stream: http://<pi-ip>:8888/
"""
import cv2, time, threading, signal, subprocess, struct
import numpy as np
import serial
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket as _socket
from ultralytics import YOLO

# ============ CONFIG ============
MODEL_PATH    = "/home/pi/best_ncnn_model"
CAM_INDEX     = 0
WIDTH, HEIGHT = 640, 480
STREAM_PORT   = 8888
CONF          = 0.45    # slightly relaxed: model is very confident (0.97+) on clean balls; catches partial-occlusion frames

# Camera positions — dynamic tilt: FAR for chase+goal sighting, CLOSE for confirming ball in scoop
CAM_PAN_FIXED  = 1596     # calibrated — pan stays locked
CAM_TILT_FAR   = 1770     # default — sees ball + goal (tuned 2026-05-04)
CAM_TILT_CLOSE = 1820     # used when ball within scoop range — looks down so ball stays in frame
CAM_TILT_FIXED = CAM_TILT_FAR  # back-compat for overlay/init
TILT_NEAR_CM   = 22.0     # at FAR: switch to CLOSE when ball ≤ this
TILT_FAR_CM    = 38.0     # at CLOSE: switch back to FAR when ball > this (or ball lost)
TILT_DEBOUNCE_S = 0.5     # min interval between tilt switches

# Detection stability
EMA_ALPHA      = 0.55     # ball-position smoothing
BALL_PERSIST_N = 3        # consecutive frames before PID/body acts
BALL_TTL_N     = 20       # ~3.3s @ 6 FPS — keep last ball through brief occlusions (goal mesh, motion blur)

# Distance estimate (vision, calibrated for 40mm ball + this camera)
FOCAL_LENGTH     = 474.0
BALL_DIAMETER_MM = 40.0
def distance_cm(r):
    return 0.0 if r <= 0 else (BALL_DIAMETER_MM * FOCAL_LENGTH) / (r * 2) / 10.0

# Motor safety
NO_MOTORS         = False  # SAFE: True = wheels disabled. False = bot drives.
FWD_DUTY          = 28.0   # tuned for fresh battery (high torque) — was 36 for tired battery
TURN_DUTY         = 18.0   # gentler curving
MAX_DUTY          = 40.0
TURN_IN_PLACE_DTY = 26.0   # gentler in-place pivots; fresh battery still breaks stiction

# Body rotation pulses (long enough to overcome carpet stiction, short enough to avoid overshoot)
BODY_ROT_PULSE_S = 0.06    # smaller kick = smaller overshoot on carpet
BODY_ROT_STOP_S  = 0.28    # longer settle = YOLO sees more frames before next correction

# Forward drive pulses (controlled increments so YOLO can keep up with closing distance)
FWD_PULSE_S = 0.15         # drive forward this long
FWD_STOP_S  = 0.15         # then stop and let YOLO update (~2-3 frames at 20 FPS)

# State machine thresholds
BALL_CENTER_TIGHT = 80     # rotate->forward transition: commit when ball is "approximately" centered
BALL_CENTER_LOOSE = 280    # forward->rotate transition: stay in curving CHASE_FWD until ball drifts past frame edge
SCOOP_DIST_CM     = 12.0
SCOOP_CENTER_TOL  = 130    # ball must be within 130px of frame center to count as "in scoop" (i.e. at bot's mouth)
AIM_TOL_PX        = 100    # very generous — commit to SCORE early, avoid rotation oscillation
GOAL_CLOSE_W_PX   = 9999   # disabled: goal-width based AT_GOAL stop. Bot will AIM + SCORE instead.
SCORE_DURATION_S  = 0.9    # longer push so ball actually goes INTO goal mouth (not just bumps it)
SLOWDOWN_ERR_PX   = 250    # disabled in practice: bot transitions to CHASE_ROT at LOOSE=220 before this fires
AIM_DRIVE_LIN     = 0.55   # need higher than chase since FWD_DUTY=28 means 0.55*28=15.4 (one side ~32% — breaks stiction)
AIM_DRIVE_ANG     = 0.95   # paired with above to keep one side ~zero, other side ~32% (clean pivot+forward)

# ============ Serial / CRC8 ============
CRC8 = [
    0,94,188,226,97,63,221,131,194,156,126,32,163,253,31,65,157,195,33,127,252,162,64,30,
    95,1,227,189,62,96,130,220,35,125,159,193,66,28,254,160,225,191,93,3,128,222,60,98,
    190,224,2,92,223,129,99,61,124,34,192,158,29,67,161,255,70,24,250,164,39,121,155,197,
    132,218,56,102,229,187,89,7,219,133,103,57,186,228,6,88,25,71,165,251,120,38,196,154,
    101,59,217,135,4,90,184,230,167,249,27,69,198,152,122,36,248,166,68,26,153,199,37,123,
    58,100,134,216,91,5,231,185,140,210,48,110,237,179,81,15,78,16,242,172,47,113,147,205,
    17,79,173,243,112,46,204,146,211,141,111,49,178,236,14,80,175,241,19,77,206,144,114,44,
    109,51,209,143,12,82,176,238,50,108,142,208,83,13,239,177,240,174,76,18,145,207,45,115,
    202,148,118,40,171,245,23,73,8,86,180,234,105,55,213,139,87,9,235,181,54,104,138,212,
    149,203,41,119,244,170,72,22,233,183,85,11,136,214,52,106,43,117,151,201,74,20,246,168,
    116,42,200,150,21,75,169,247,182,232,10,84,215,137,107,53
]
try:
    SERIAL_PORT = serial.Serial("/dev/rrc", baudrate=1000000, timeout=5)
    SERIAL_PORT.rts = False; SERIAL_PORT.dtr = False
    print("[serial] /dev/rrc opened")
except Exception as e:
    print(f"[warn] serial open failed: {e}")
    SERIAL_PORT = None

def _send(func, data):
    if SERIAL_PORT is None: return
    buf = [0xAA, 0x55, func, len(data)] + list(data)
    c = 0
    for b in buf[2:]: c = CRC8[c ^ b]
    buf.append(c & 0xFF)
    try: SERIAL_PORT.write(bytes(buf))
    except Exception: pass

def servo_write(positions, duration=0.02):
    ms = int(duration * 1000)
    data = [0x01, ms & 0xFF, (ms >> 8) & 0xFF, len(positions)]
    for sid, pos in positions:
        data.extend(struct.pack("<BH", sid, int(pos)))
    _send(4, data)

def motor_duty(duties):
    data = [0x05, len(duties)]
    for mid, v in duties:
        data.extend(struct.pack("<Bf", int(mid - 1), float(v)))
    _send(3, data)

def stop_motors():
    motor_duty([[1, 0], [2, 0], [3, 0], [4, 0]])

_last_drive = [0.0, (0.0, 0.0)]
def drive(linear_x, angular_z):
    if NO_MOTORS: return
    now = time.time()
    cmd = (round(linear_x, 2), round(angular_z, 2))
    if now - _last_drive[0] < 0.2 and cmd == _last_drive[1]:
        return
    _last_drive[0] = now; _last_drive[1] = cmd
    fwd = float(np.clip(linear_x,  -1.0, 1.0)) * FWD_DUTY
    tur = float(np.clip(angular_z, -1.0, 1.0)) * TURN_DUTY
    m1 = -fwd + tur
    m2 =  fwd + tur
    m3 = -fwd + tur
    m4 =  fwd + tur
    clamp = lambda v: float(max(-MAX_DUTY, min(MAX_DUTY, v)))
    motor_duty([[1, clamp(m1)], [2, clamp(m2)], [3, clamp(m3)], [4, clamp(m4)]])

# Pulsed body rotation — very short pulses so each commit is small (3-5 deg per pulse)
_body_rot_phase = 'STOP'
_body_rot_phase_start = 0.0
def body_rot_pulsed(direction):
    """Pulse-rotate. direction: +1 = CCW (left), -1 = CW (right). Caller invokes every loop."""
    if NO_MOTORS: return
    global _body_rot_phase, _body_rot_phase_start
    now = time.time()
    elapsed = now - _body_rot_phase_start
    if _body_rot_phase == 'ROTATE':
        if elapsed >= BODY_ROT_PULSE_S:
            _body_rot_phase = 'STOP'; _body_rot_phase_start = now
            stop_motors()
        else:
            duty = TURN_IN_PLACE_DTY * direction
            motor_duty([[1, duty], [2, duty], [3, duty], [4, duty]])
    else:  # STOP
        if elapsed >= BODY_ROT_STOP_S:
            _body_rot_phase = 'ROTATE'; _body_rot_phase_start = now
            duty = TURN_IN_PLACE_DTY * direction
            motor_duty([[1, duty], [2, duty], [3, duty], [4, duty]])
        else:
            stop_motors()

def reset_body_rot_phase():
    global _body_rot_phase, _body_rot_phase_start
    _body_rot_phase = 'STOP'
    _body_rot_phase_start = time.time()

# Pulsed forward drive — drive briefly, stop briefly, repeat. Lets YOLO keep up with closing distance.
_fwd_phase = 'STOP'
_fwd_phase_start = 0.0
def chase_fwd_pulsed(lin, ang=0.0):
    """Pulsed forward drive with optional angular curve toward ball. Caller invokes every loop iter."""
    if NO_MOTORS: return
    global _fwd_phase, _fwd_phase_start
    now = time.time()
    elapsed = now - _fwd_phase_start
    if _fwd_phase == 'DRIVE':
        if elapsed >= FWD_PULSE_S:
            _fwd_phase = 'STOP'; _fwd_phase_start = now
            stop_motors()
        else:
            drive(lin, ang)
    else:  # STOP
        if elapsed >= FWD_STOP_S:
            _fwd_phase = 'DRIVE'; _fwd_phase_start = now
            drive(lin, ang)
        else:
            stop_motors()

def reset_fwd_phase():
    global _fwd_phase, _fwd_phase_start
    _fwd_phase = 'STOP'
    _fwd_phase_start = time.time()

# ============ camera + model ============
print(f"loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")
print("model loaded")

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# minimize capture-side latency: 1-frame V4L2 buffer + MJPG input fourcc
# (without these, cap.read() returns the OLDEST buffered frame, which adds
# 200-400ms of pipeline lag in front of our inference time)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
except Exception:
    pass
if not cap.isOpened():
    print("ERROR: camera"); raise SystemExit(1)
for _ in range(8): cap.read()

# Set initial camera position — pan locked, tilt is dynamic (FAR by default)
stop_motors()
servo_write([[1, CAM_TILT_FAR], [2, CAM_PAN_FIXED]], 0.5)
time.sleep(0.6)
print(f"[servo] init pan={CAM_PAN_FIXED} tilt={CAM_TILT_FAR} (FAR)")
print(f"[motors] {'DISARMED' if NO_MOTORS else 'ARMED'}")

# ============ stability state ============
last_ball       = None    # (cx, cy, w, h) smoothed + persistent
ball_smooth     = None    # (cx, cy)
ball_persist    = 0
ball_ttl        = 0
chase_rotating  = True    # hysteresis: True = currently in CHASE_ROT
score_started_t = 0.0

# Dynamic tilt state — camera dips down when ball is close so it doesn't drop out of frame
_cam_state      = "FAR"   # "FAR" or "CLOSE"
_cam_switch_t   = 0.0
_last_goal_x    = None    # remembered from last FAR view; used when at CLOSE (goal not visible)
_last_goal_w    = 0       # remembered goal bbox width — for goal_close check at CLOSE tilt
_last_goal_seen_t = 0.0
def update_cam_tilt(ball_dist_cm, ball_seen_now):
    """Switch tilt between FAR and CLOSE based on ball proximity. Debounced."""
    global _cam_state, _cam_switch_t
    now = time.time()
    if now - _cam_switch_t < TILT_DEBOUNCE_S:
        return
    if _cam_state == "FAR":
        if ball_seen_now and ball_dist_cm <= TILT_NEAR_CM:
            _cam_state = "CLOSE"; _cam_switch_t = now
            stop_motors()  # halt during tilt — verify scoop before resuming
            servo_write([[1, CAM_TILT_CLOSE], [2, CAM_PAN_FIXED]], 0.2)
            print(f"[cam] FAR→CLOSE (ball at {ball_dist_cm:.1f}cm) — verifying scoop")
    else:  # CLOSE
        # leave CLOSE when ball gets far again, OR ball lost for >2s (re-acquire from FAR)
        if (ball_seen_now and ball_dist_cm > TILT_FAR_CM) or (not ball_seen_now and (now - _cam_switch_t) > 2.0):
            _cam_state = "FAR"; _cam_switch_t = now
            stop_motors()
            servo_write([[1, CAM_TILT_FAR], [2, CAM_PAN_FIXED]], 0.2)
            print(f"[cam] CLOSE→FAR")

# ============ stream + sys stats ============
state_pub = {"frame": None, "frame_id": 0, "fps": 0.0, "running": True,
             "temp": "?", "volts": "?", "throttled": "0x0"}
lock = threading.Lock()

CLASS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255)}
CLASS_NAMES  = {0: "ball", 1: "goalpost"}

def handle_sigint(sig, frm):
    print("[sigint] stopping")
    state_pub["running"] = False
    try: stop_motors()
    except Exception: pass
    try: servo_write([[1, 1500], [2, 1500]], 0.5)
    except Exception: pass
signal.signal(signal.SIGINT,  handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

def poll_stats():
    while state_pub["running"]:
        try:
            t  = subprocess.check_output(["vcgencmd","measure_temp"],         text=True).strip()
            v  = subprocess.check_output(["vcgencmd","measure_volts","core"], text=True).strip()
            th = subprocess.check_output(["vcgencmd","get_throttled"],        text=True).strip()
            state_pub["temp"]      = t.split("=")[1] if "=" in t else "?"
            state_pub["volts"]     = v.split("=")[1] if "=" in v else "?"
            state_pub["throttled"] = th.split("=")[1] if "=" in th else "?"
        except Exception:
            pass
        time.sleep(2.0)
threading.Thread(target=poll_stats, daemon=True).start()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
            self.wfile.write(b'<html><body style="background:#111;color:#fff;font-family:monospace;text-align:center">'
                             b'<h2>YOLO striker - LOCKED CAMERA</h2><img src="/stream"/></body></html>'); return
        if self.path != "/stream":
            self.send_response(404); self.end_headers(); return
        # low-latency stream: TCP_NODELAY + tiny send buffer so old frames are dropped
        # by the kernel rather than queued, and per-frame deduplication so we never
        # encode/transmit the same frame twice.
        try:
            self.connection.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            self.connection.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 65536)
        except OSError:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        last_id = -1
        try:
            while state_pub["running"]:
                with lock:
                    fid = state_pub["frame_id"]
                    f = None if state_pub["frame"] is None or fid == last_id else state_pub["frame"]
                if f is None:
                    time.sleep(0.01); continue
                last_id = fid
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if not ok: continue
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

def serve():
    HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler).serve_forever()
threading.Thread(target=serve, daemon=True).start()
print(f"streaming: http://<pi-ip>:{STREAM_PORT}/")

# ============ main loop ============
fcx = WIDTH // 2; fcy = HEIGHT // 2
t_last, frames, fps_acc = time.time(), 0, 0.0
mot_state = "INIT"

try:
    while state_pub["running"]:
        # drain any frames the V4L2 driver buffered while we were doing inference,
        # then keep the most recent one — costs ~1-2ms and removes pipeline lag
        cap.grab(); cap.grab()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue

        t0 = time.time()
        res = model.predict(frame, imgsz=640, conf=CONF, verbose=False, device="cpu")[0]
        infer_ms = (time.time() - t0) * 1000

        # ----- detect ball + goal -----
        best_ball = None; best_ball_a = 0
        best_goal = None; best_goal_a = 0
        best_goal_clipped = False
        n_ball = n_goal = 0
        EDGE_MARGIN = 10  # px: bbox touching this close to a frame edge is "clipped"
        fw_full = frame.shape[1]
        for box in res.boxes:
            cls = int(box.cls[0]); conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1; h = y2 - y1; a = w * h
            cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
            color = CLASS_COLORS.get(cls, (255,255,255))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{CLASS_NAMES.get(cls,'?')} {conf:.2f}", (x1, max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
            cv2.circle(frame, (int(cx), int(cy)), 8, (255,255,255), 1)
            cv2.line(frame, (int(cx), y1), (int(cx), y2), color, 1)
            if cls == 0:
                n_ball += 1
                if a > best_ball_a:
                    best_ball = (cx, cy, w, h); best_ball_a = a
            elif cls == 1:
                n_goal += 1
                if a > best_goal_a:
                    best_goal = (cx, cy, w, h); best_goal_a = a
                    best_goal_clipped = (x1 <= EDGE_MARGIN) or (x2 >= fw_full - EDGE_MARGIN)
        # mark clipped goal bbox visually so user can see when goal_x is unreliable
        if best_goal is not None and best_goal_clipped:
            cv2.putText(frame, "GOAL CLIPPED", (int(best_goal[0])-90, int(best_goal[1])-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ----- EMA smoothing + persistence + TTL on ball -----
        if best_ball is not None:
            fcx_raw, fcy_raw, fw, fh = best_ball
            if ball_smooth is None:
                ball_smooth = (fcx_raw, fcy_raw)
            else:
                ball_smooth = (
                    EMA_ALPHA * fcx_raw + (1 - EMA_ALPHA) * ball_smooth[0],
                    EMA_ALPHA * fcy_raw + (1 - EMA_ALPHA) * ball_smooth[1],
                )
            last_ball    = (ball_smooth[0], ball_smooth[1], fw, fh)
            ball_ttl     = BALL_TTL_N
            ball_persist = min(ball_persist + 1, BALL_PERSIST_N + 5)
        elif ball_ttl > 0:
            ball_ttl -= 1
        else:
            last_ball    = None
            ball_smooth  = None
            ball_persist = 0

        ball_actionable = (last_ball is not None) and (ball_persist >= BALL_PERSIST_N)

        # ----- remember goal X+W (any cam state) but ONLY when bbox is fully in frame -----
        # When the goal bbox is clipped at a frame edge its center is biased toward the visible
        # half, so we don't trust it for memory or aim — keep using the last unclipped read.
        if best_goal is not None and not best_goal_clipped:
            _last_goal_x = float(best_goal[0])
            _last_goal_w = int(best_goal[2])
            _last_goal_seen_t = time.time()

        # ----- update camera tilt based on ball distance -----
        if last_ball is not None:
            _ball_radius_px_for_tilt = min(last_ball[2], last_ball[3]) / 2.0
            update_cam_tilt(distance_cm(_ball_radius_px_for_tilt), best_ball is not None)
        else:
            update_cam_tilt(999.0, False)

        # ===== STATE MACHINE (dynamic tilt) =====
        # During the first 0.4s after a tilt change, motors stay STOPPED so YOLO can
        # re-acquire the ball at the new tilt before any drive/rotate decisions.
        tilt_settling = (time.time() - _cam_switch_t) < 0.4

        # Pre-compute goal_close from any source we have (live or remembered)
        _goal_seen_now_for_safety = best_goal is not None
        if _goal_seen_now_for_safety:
            _safety_goal_w = int(best_goal[2])
        elif _last_goal_x is not None and (time.time() - _last_goal_seen_t) < 2.0:
            _safety_goal_w = int(_last_goal_w)
        else:
            _safety_goal_w = 0

        if NO_MOTORS:
            mot_state = "DISARMED"
            mot_color = (128, 128, 128)
            stop_motors()       # safety: zero motors every loop in disarmed mode
        elif tilt_settling:
            mot_state = f"VERIFY ({_cam_state.lower()})"
            mot_color = (200, 200, 0)
            reset_body_rot_phase(); reset_fwd_phase()
            stop_motors()
        elif _safety_goal_w > GOAL_CLOSE_W_PX:
            # SAFETY: goal is huge in frame — bot is at/inside the goal. Stop unconditionally.
            mot_state = "AT_GOAL"; mot_color = (0, 220, 0)
            reset_body_rot_phase(); reset_fwd_phase()
            stop_motors()
        elif not ball_actionable:
            mot_state = "WAIT" if last_ball is None else "WARMUP"
            mot_color = (200, 200, 0)
            reset_body_rot_phase()
            stop_motors()
        else:
            ball_radius_px = min(last_ball[2], last_ball[3]) / 2.0
            ball_dist_cm   = distance_cm(ball_radius_px)
            ball_err_x     = last_ball[0] - fcx          # >0: ball right of center

            # Hysteresis: tight to commit to forward, loose to fall back to rotate
            if chase_rotating:
                centered = abs(ball_err_x) < BALL_CENTER_TIGHT
            else:
                centered = abs(ball_err_x) < BALL_CENTER_LOOSE
            chase_rotating = not centered

            # in_scoop needs TIGHTER centering than chase: ball must be at bot's mouth (frame center),
            # not just within the loose chase tolerance. Otherwise AIM aligns to off-center ball.
            in_scoop = ball_dist_cm <= SCOOP_DIST_CM and abs(ball_err_x) < SCOOP_CENTER_TOL
            # Goal info: prefer current frame UNCLIPPED; if live is clipped or missing, fall back to memory.
            # Clipped live frames have a biased center, so trust the last unclipped read instead.
            goal_seen_now = best_goal is not None and not best_goal_clipped
            if goal_seen_now:
                goal_x  = float(best_goal[0])
                goal_w_px = int(best_goal[2])
                goal_known = True
            elif _last_goal_x is not None and (time.time() - _last_goal_seen_t) < 4.0:
                goal_x = _last_goal_x
                goal_w_px = _last_goal_w
                goal_known = True
            else:
                goal_x = 0.0
                goal_w_px = 0
                goal_known = False
            aim_err   = (last_ball[0] - goal_x) if goal_known else 0.0
            aimed     = goal_known and abs(aim_err) < AIM_TOL_PX
            goal_close = goal_known and goal_w_px > GOAL_CLOSE_W_PX

            # 4-tier distance speed profile (zhui's proven values)
            if   ball_dist_cm <= 12: chase_lin = 0.78
            elif ball_dist_cm <= 25: chase_lin = 0.88
            elif ball_dist_cm <= 45: chase_lin = 0.78
            else:                    chase_lin = 0.78
            if abs(ball_err_x) > SLOWDOWN_ERR_PX:
                chase_lin *= 0.7

            if in_scoop and aimed and goal_close:
                # Bot is at goal — ball is essentially scored, don't ram the post.
                mot_state = "AT_GOAL"; mot_color = (0, 220, 0)
                stop_motors()
            elif in_scoop and aimed:
                if mot_state != "SCORE":
                    score_started_t = time.time()
                if time.time() - score_started_t < SCORE_DURATION_S:
                    mot_state = "SCORE"; mot_color = (0, 255, 0)
                    drive(1.0, 0.0)
                else:
                    mot_state = "SCORE_DONE"; mot_color = (0, 200, 0)
                    stop_motors()
            elif in_scoop and goal_known:
                # Curving AIM: drive forward while rotating. Pure in-place rotation can't
                # swing the goal across frame fast enough when ball is in scoop (ball moves
                # with bot). Forward motion changes geometry so rotation actually converges.
                mot_state = "AIM (curve)"
                mot_color = (0, 255, 255)
                reset_body_rot_phase(); reset_fwd_phase()
                # aim_err > 0: ball right of goal → rotate left (+1) to swing ball toward goal
                direction = +1 if aim_err > 0 else -1
                drive(AIM_DRIVE_LIN, AIM_DRIVE_ANG * direction)
            elif in_scoop:
                mot_state = "HOLD"; mot_color = (255, 200, 0)
                stop_motors()
            elif centered:
                mot_state = f"CHASE_FWD ({_fwd_phase.lower()})"
                mot_color = (255, 200, 0)
                reset_body_rot_phase()
                # Curving chase: small angular proportional to ball offset so bot leans toward ball
                # while driving forward, instead of straight forward and rotating separately.
                chase_ang = float(np.clip(-ball_err_x / 600.0, -0.5, 0.5))
                chase_fwd_pulsed(chase_lin, chase_ang)
            else:
                mot_state = f"CHASE_ROT ({_body_rot_phase.lower()})"
                mot_color = (255, 200, 0)
                reset_fwd_phase()
                # ball_err_x > 0 (right of center) → rotate body RIGHT (CW = -1)
                direction = -1 if ball_err_x > 0 else +1
                body_rot_pulsed(direction)

        # ----- aim line: ball -> goal -----
        if best_ball is not None and best_goal is not None:
            cv2.line(frame,
                     (int(best_ball[0]), int(best_ball[1])),
                     (int(best_goal[0]), int(best_goal[1])),
                     (0, 255, 255), 1)

        # ----- crosshair (camera frame center, since camera is locked) -----
        cv2.line(frame, (fcx, 0), (fcx, frame.shape[0]), (40, 40, 200), 1)
        cv2.line(frame, (fcx-25, fcy), (fcx+25, fcy), (0, 0, 255), 2)
        cv2.line(frame, (fcx, fcy-25), (fcx, fcy+25), (0, 0, 255), 2)
        # ----- scoop tolerance band (shows where ball must be to trigger AIM) -----
        cv2.line(frame, (fcx-SCOOP_CENTER_TOL, 0), (fcx-SCOOP_CENTER_TOL, frame.shape[0]), (0, 200, 200), 1)
        cv2.line(frame, (fcx+SCOOP_CENTER_TOL, 0), (fcx+SCOOP_CENTER_TOL, frame.shape[0]), (0, 200, 200), 1)

        # ----- FPS / overlay -----
        frames += 1
        if frames >= 5:
            fps_acc = frames / (time.time() - t_last)
            t_last, frames = time.time(), 0

        thr = state_pub["throttled"]
        thr_color = (0, 255, 0) if thr == "0x0" else (0, 0, 255)
        ball_x = int(last_ball[0]) if last_ball else 0
        if best_goal is not None and not best_goal_clipped:
            goal_x_o = int(best_goal[0])
            goal_w_o = int(best_goal[2])
            goal_label = "live"
        elif _last_goal_x is not None and (time.time() - _last_goal_seen_t) < 4.0:
            goal_x_o = int(_last_goal_x)
            goal_w_o = int(_last_goal_w)
            goal_label = f"mem {time.time()-_last_goal_seen_t:.1f}s" + (" (clipped)" if best_goal is not None else "")
        else:
            goal_x_o = 0
            goal_w_o = 0
            goal_label = "none"
        if last_ball is not None:
            r_px = min(last_ball[2], last_ball[3]) / 2.0
            dist_str = f"{distance_cm(r_px):5.1f} cm  (r={int(r_px)}px)"
        else:
            dist_str = "—"
        motors_label = "OFF" if NO_MOTORS else "ON"
        cam_tilt_now = CAM_TILT_CLOSE if _cam_state == "CLOSE" else CAM_TILT_FAR
        lines = [
            (f"BODY: {mot_state}  |  motors: {motors_label}",                mot_color),
            (f"FPS: {fps_acc:5.1f}  infer: {infer_ms:5.1f}ms",               (255, 255, 255)),
            (f"ball: {n_ball} (x={ball_x})  dist: {dist_str}",               (0, 255, 200)),
            (f"goal: {goal_label} (x={goal_x_o}, w={goal_w_o}px)",           (255, 255, 255)),
            (f"camera {_cam_state}  pan={CAM_PAN_FIXED} tilt={cam_tilt_now}",(180, 180, 180)),
            (f"temp: {state_pub['temp']}  volts: {state_pub['volts']}",      (255, 255, 255)),
            (f"throttled: {thr}",                                            thr_color),
        ]
        y = 25
        for line, color in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 4)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 22

        with lock:
            state_pub["frame"] = frame
            state_pub["frame_id"] += 1
            state_pub["fps"]   = fps_acc

finally:
    state_pub["running"] = False
    try: stop_motors()
    except Exception: pass
    try: servo_write([[1, 1500], [2, 1500]], 0.5)
    except Exception: pass
    cap.release()
    print("done")
