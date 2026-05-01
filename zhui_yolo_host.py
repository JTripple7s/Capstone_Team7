#!/usr/bin/env python3
"""zhui_yolo_host.py — Kevin's zhui_yolo.py adapted to run on host (no ROS, no docker).
Same ball-chase logic, NCNN backend, no goal-seeking, pulsed rotation during search.

Source: /home/pi/zhui_yolo.py (Kevin's pre-friend tracker)
Changes: ROS Image subscriber → cv2.VideoCapture; path → /home/pi/Bestv2_ncnn_model
"""
import os, sys, time, struct, threading, subprocess, signal
import http.server, socketserver
import numpy as np
import cv2
import serial

# Offline mode + thread caps
os.environ['YOLO_OFFLINE']     = '1'
os.environ['YOLO_AUTOINSTALL'] = 'False'
os.environ.setdefault('OMP_NUM_THREADS',      '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS',      '3')

# ============ CONFIG ============
MODEL_PATH    = '/home/pi/V5_ncnn_model'   # V5 (Apr 29, newest) — yolov8n with robot+ball+goalpost classes
YOLO_IMGSZ    = 320
YOLO_CONF     = 0.10                            # very low base — let everything through, filter per-class below
BALL_CONF     = 0.45                            # was 0.85 → 0.55 → 0.45. HSV + tiered persistence catch false positives at this threshold.
GOAL_CONF     = 0.20                            # goalpost: lenient
BALL_MIN_RADIUS_PX = 18                         # was 35 — too strict, missed far balls. Tiered persistence below handles false-positive risk.
BALL_PERSIST_FRAMES = 3                         # frames needed before CARRYING transition
CHASE_PERSIST_FRAMES = 3                        # base frames for close balls; far balls require more (see ball_actionable below)
BALL_MAX_JUMP_PX = 180                          # reject ball detection that teleports between frames
BALL_PERSIST_FAR_FRAMES = 8                     # tiered: small/far balls (radius < BALL_RADIUS_FAR_THRESHOLD) need more frames
BALL_RADIUS_FAR_THRESHOLD = 25                  # below this radius, treat as "far ball" → stricter persistence
CARRYING_LOST_BAIL_S = 2.0                      # if ball not seen for 2s while CARRYING, bail to HUNTING
FIXED_TILT       = 1850                         # tilt during CARRYING (no sweep there)
# 3 search tilt layers: low → mid → high (camera physically tilts up between layers)
# Higher tilt # = camera looks DOWN. So layer 0 (2000) = closer/down, layer 2 (1700) = far/up.
SEARCH_TILT_LEVELS = [2000, 1850, 1700]
SEARCH_TURN_DURATION_S = 1.6                    # ~90° at 18% duty (slower duty needs longer time)
SEARCH_TURN_RATE       = 0.65                   # angular rate during TURN phase (carpet)
DEBUG_RENDER_ALL = False                        # OFF
NO_MOTORS        = False                        # set True for SAFE MODE (servos only, no wheels)
YOLO_EVERY_N  = 1
BALL_TOP_MARGIN = 0.20                          # ignore detections in top 20% of frame (ceiling area)
CAMERA_DEV    = '/dev/video0'
CAM_W, CAM_H  = 640, 480
MJPEG_PORT    = 8888

# ============ Mission state machine (ball-to-goal) ============
# Bot has front scoop (2 poles + curved cable cradle) — ball gets captured on contact.
# So instead of STOPPING at the ball, we keep ball pinned in scoop while scanning for goal.
CARRY_DIST_CM      = 12.0      # at this distance, ball is in the scoop -> CARRYING
GOAL_SCAN_TILTS    = [1700, 1850, 2000]   # 3 layers: high, mid, low
GOAL_SCAN_LAYER_S  = 1.5       # seconds per tilt layer
GOAL_SCAN_TIMEOUT  = 12.0      # give up after this and return to HUNTING
GOAL_CENTERED_PX   = 60        # |goal_ex| under this = centered enough to score
PUSH_DURATION_S    = 2.5       # how long to drive forward in SCORING
CARRY_FWD          = 0.50      # firm forward push to drive ball forward
CARRY_TURN         = 0.0       # NO rotation during carry — was causing bot to spin in place

FOCAL_LENGTH     = 474.0
BALL_DIAMETER_MM = 40.0

def distance_cm(radius):
    if radius <= 0: return 0
    return (BALL_DIAMETER_MM * FOCAL_LENGTH) / (radius * 2) / 10.0

# ============ Serial protocol (same as zhui_yolo.py) ============
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
except Exception as e:
    print(f"[warn] serial open failed: {e}", flush=True)
    SERIAL_PORT = None

def _send(func, data):
    if SERIAL_PORT is None: return
    buf = [0xAA, 0x55, func, len(data)] + list(data)
    c = 0
    for b in buf[2:]:
        c = CRC8[c ^ b]
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

_last_drive_time = [0.0]
_last_drive_cmd  = [(0.0, 0.0)]

def drive(linear_x, angular_z):
    """Tuned for carpet — needs >=30% duty to overcome static friction."""
    if NO_MOTORS:
        return
    FWD_DUTY  = 20.0   # slow pace — was 34 for carpet, then 24, now 20 for smooth hardwood
    TURN_DUTY = 14.0
    MAX_ANY   = 24.0
    now = time.time()
    cmd = (round(linear_x, 2), round(angular_z, 2))
    if now - _last_drive_time[0] < 0.2 and cmd == _last_drive_cmd[0]:
        return
    _last_drive_time[0] = now
    _last_drive_cmd[0] = cmd
    fwd = np.clip(linear_x,  -1.0, 1.0) * FWD_DUTY
    tur = np.clip(angular_z, -1.0, 1.0) * TURN_DUTY
    m1 = -fwd + tur
    m2 =  fwd + tur
    m3 = -fwd + tur
    m4 =  fwd + tur
    clamp = lambda v: float(max(-MAX_ANY, min(MAX_ANY, v)))
    motor_duty([[1, clamp(m1)], [2, clamp(m2)], [3, clamp(m3)], [4, clamp(m4)]])

def stop_motors():
    motor_duty([[1, 0], [2, 0], [3, 0], [4, 0]])

def turn_in_place(direction):
    """Pure-rotation. Reduced duty for hardwood — was 32% for carpet.
    direction = +1 for clockwise, -1 for counter-clockwise.
    """
    if NO_MOTORS:
        return
    duty = 18.0 * direction   # slowed: was 32 carpet → 22 → 18 hardwood
    motor_duty([[1, duty], [2, duty], [3, duty], [4, duty]])

# ============ HSV green-ball verifier (rejects YOLO false positives) ============
# Tennis-ball / green-ball hue ~ 35-85 (OpenCV HSV: H 0-179).
GREEN_HSV_LO = np.array([30,  60,  60], dtype=np.uint8)
GREEN_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)
GREEN_FRAC_MIN = 0.18   # >=18% of bbox must be green to accept

def is_green_ball(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    crop = frame[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)
    frac = float(mask.mean()) / 255.0
    return (frac >= GREEN_FRAC_MIN), frac

# ============ MJPEG server ============
latest_frame = None
frame_lock = threading.Lock()

class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body style="margin:0;background:#000">'
                             b'<img src="/video" style="width:100%"></body></html>')
        elif self.path == '/video':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        f = latest_frame
                    if f is not None:
                        ok, jpg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                                             + jpg.tobytes() + b'\r\n')
                    time.sleep(0.033)
            except Exception:
                pass
        else:
            self.send_error(404)

class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

def start_mjpeg(port=MJPEG_PORT):
    srv = ReusableTCPServer(("", port), MJPEGHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

# ============ Tracker ============
class ZhuiYolo:
    def __init__(self, model):
        self.model = model
        # Accept any class with 'ball' in name. If none, fall back to all.
        self.ball_class_ids = set(
            cid for cid, n in self.model.names.items() if 'ball' in str(n).lower()
        )
        if not self.ball_class_ids:
            self.ball_class_ids = set(self.model.names.keys())
        print(f"[tracker] ball class ids: {self.ball_class_ids}", flush=True)

        self.pan  = 1500.0
        self.tilt = 1800.0
        self.pan_i = 0.0;  self.tilt_i = 0.0
        self.pan_last = 0.0; self.tilt_last = 0.0
        # PID gains — slowed from 0.25/0.05/0.009 for smoother camera tracking.
        self.Kp, self.Ki, self.Kd = 0.15, 0.03, 0.006
        servo_write([[1, int(self.tilt)], [2, int(self.pan)]], 0.5)

        self.search_dir = 1     # pan direction: +1 = left→right, -1 = right→left
        self.search_phase = 0   # tilt level index
        self.tilt_dir = 1       # tilt direction: +1 = phase++ (going UP visually), -1 = going DOWN
        self.lost = 0
        self.frame_counter = 0
        self.last_ball = None
        self.last_goal = None
        # Stability state: TTL (keep last detection for a few miss-frames) + EMA smoothing
        self.ball_ttl = 0
        self.goal_ttl = 0
        self.ball_smooth = None     # EMA-smoothed (cx, cy, r)
        self.goal_smooth = None     # EMA-smoothed (cx, cy, w, h)
        # Mission state machine: HUNTING -> AT_BALL_SCAN -> ALIGNED -> SCORING -> HUNTING
        self.mission = "HUNTING"
        self.scan_start = 0.0
        self.scan_dir   = 1          # body rotation direction during scan
        self.score_start = 0.0
        self.ball_persist = 0        # consecutive frames with ball detected (gate for CARRYING)
        # Search sub-state: alternates between PAN (camera sweep) and TURN (body rotates 90°)
        self.search_sub = 'PAN'
        self.turn_start_time = 0.0
        self.search_layer = 0          # current tilt layer (0 = low, 2 = high)
        self.search_dir = -1           # start R→L (pan from high → low)
        # Goalpost bearing memory: remember which side we last saw the goal (left/right).
        # Used in CARRYING when goal not visible — bias rotation toward last-known direction.
        self.last_goal_seen_t = 0.0
        self.last_goal_side  = 0       # -1 = left, +1 = right, 0 = unknown
        # Reject-reason diagnostic: log every N seconds when raw ball detections were rejected.
        self.last_reject_log_t = 0.0
        self.last_ball_raw = 0
        self.last_ball_reject = {}

    def run_yolo(self, frame):
        fh, fw = frame.shape[:2]
        results = self.model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                                     verbose=False, device='cpu')
        if not results:
            return None, None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None, None
        best_ball = None; best_ball_area = 0
        best_goal = None; best_goal_area = 0
        # Debug: count what model returns per class (helps diagnose "model never sees goalpost")
        cls_count = {}
        all_detections = []  # for debug-render-all
        # Diagnostic: count why ball candidates got rejected (helps tune filters)
        ball_reject = {'conf': 0, 'top': 0, 'selfmask': 0, 'ratio': 0, 'radius': 0, 'hsv': 0}
        ball_raw    = 0  # how many raw ball detections the model produced (any conf)
        for box in r.boxes:
            c = int(box.cls[0])
            cls_count[c] = cls_count.get(c, 0) + 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            all_detections.append({
                'cls': c, 'name': self.model.names[c],
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'conf': float(box.conf[0]),
            })
        self.last_cls_count = cls_count
        self.last_all = all_detections
        self.last_ball_raw = ball_raw
        self.last_ball_reject = ball_reject
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1; h = y2 - y1
            a = w * h
            if a < 100: continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            ratio = w / h if h > 0 else 0
            name = self.model.names[cls]

            conf = float(box.conf[0])
            if cls in self.ball_class_ids:
                ball_raw += 1
                if conf < BALL_CONF:
                    ball_reject['conf'] += 1; continue
                # ball-specific filters
                if cy < fh * BALL_TOP_MARGIN:
                    ball_reject['top'] += 1; continue
                # ROBOT SELF-MASK: bottom-left + bottom-right corners (own wheels).
                # Tightened to corners only (fw*0.15 / 0.85) so balls in normal-bottom area aren't rejected.
                if cy > fh * 0.85:
                    if cx < fw * 0.15 or cx > fw * 0.85:
                        ball_reject['selfmask'] += 1; continue
                if ratio < 0.5 or ratio > 1.8:
                    ball_reject['ratio'] += 1; continue
                # Size filter — reject tiny detections (most false positives are small)
                radius_check = min(w, h) / 2.0
                if radius_check < BALL_MIN_RADIUS_PX:
                    ball_reject['radius'] += 1; continue
                # HSV green-color verifier: rejects YOLO false positives that aren't actually green.
                ok_green, gfrac = is_green_ball(frame, (int(x1), int(y1), int(x2), int(y2)))
                if not ok_green:
                    ball_reject['hsv'] += 1; continue
                if a > best_ball_area:
                    radius = min(w, h) / 2.0
                    col = (0, 0, 255) if 'red' in str(name).lower() else (0, 255, 0)
                    best_ball = {'cx':cx, 'cy':cy, 'r':radius, 'a':a, 'name':name.upper(),
                                 'col':col, 'conf': conf, 'gfrac': gfrac,
                                 'box': (int(x1), int(y1), int(x2), int(y2))}
                    best_ball_area = a
            elif 'goal' in name.lower():
                if conf < GOAL_CONF:
                    continue
                # goalpost: pick LARGEST detection (closest goalpost dominates)
                if a > best_goal_area:
                    best_goal = {'cx':cx, 'cy':cy, 'w':w, 'h':h, 'a':a, 'name':name.upper(),
                                 'conf': conf, 'ratio': ratio,
                                 'box': (int(x1), int(y1), int(x2), int(y2))}
                    best_goal_area = a
        return best_ball, best_goal

    def step(self, frame):
        global latest_frame
        h, w = frame.shape[:2]
        fx, fy = w // 2, h // 2
        out = frame.copy()

        self.frame_counter += 1
        if self.frame_counter % YOLO_EVERY_N == 0:
            fresh_ball, fresh_goal = self.run_yolo(frame)
        else:
            fresh_ball = fresh_goal = None

        # ----- Ball: TTL persistence + position consistency check -----
        BALL_TTL = 8        # ~1.25s at 6.4fps — bridges short YOLO drop-outs from motion blur
        if fresh_ball is not None:
            jumped = False
            if self.last_ball is not None and self.ball_ttl > 0:
                dx = fresh_ball['cx'] - self.last_ball['cx']
                dy = fresh_ball['cy'] - self.last_ball['cy']
                jump_dist = (dx*dx + dy*dy) ** 0.5
                if jump_dist > BALL_MAX_JUMP_PX:
                    jumped = True
                    print(f"[ball] REJECT JUMP {jump_dist:.0f}px (was {self.last_ball['cx']:.0f},{self.last_ball['cy']:.0f} -> {fresh_ball['cx']:.0f},{fresh_ball['cy']:.0f}) conf={fresh_ball['conf']:.2f}", flush=True)
                    self.ball_persist = 0
            if not jumped:
                # EWMA smoothing on (cx, cy, r) — reduces jitter in chase + cuts JUMP rejects.
                a = 0.35   # 0.35 new + 0.65 old
                if self.ball_smooth is None:
                    self.ball_smooth = (fresh_ball['cx'], fresh_ball['cy'], fresh_ball['r'])
                else:
                    self.ball_smooth = (
                        a * fresh_ball['cx'] + (1 - a) * self.ball_smooth[0],
                        a * fresh_ball['cy'] + (1 - a) * self.ball_smooth[1],
                        a * fresh_ball['r']  + (1 - a) * self.ball_smooth[2],
                    )
                fresh_ball['cx'] = self.ball_smooth[0]
                fresh_ball['cy'] = self.ball_smooth[1]
                fresh_ball['r']  = self.ball_smooth[2]
                # Log first detection (acquiring) and chase trigger
                if self.last_ball is None:
                    print(f"[ball] ACQUIRE conf={fresh_ball['conf']:.2f} r={fresh_ball['r']:.0f} pos=({fresh_ball['cx']:.0f},{fresh_ball['cy']:.0f}) gfrac={fresh_ball.get('gfrac', 0):.2f}", flush=True)
                self.last_ball = fresh_ball
                self.ball_ttl = BALL_TTL
                self.ball_persist += 1
                # Tiered: far balls (small radius) need more persistence frames before chasing.
                needed = BALL_PERSIST_FAR_FRAMES if fresh_ball['r'] < BALL_RADIUS_FAR_THRESHOLD else CHASE_PERSIST_FRAMES
                if self.ball_persist == needed:
                    print(f"[ball] CHASE-OK persist={self.ball_persist}/{needed} (r={fresh_ball['r']:.0f}) conf={fresh_ball['conf']:.2f}", flush=True)
            else:
                if self.ball_ttl > 0:
                    self.ball_ttl -= 1
        elif self.ball_ttl > 0:
            self.ball_ttl -= 1
        else:
            if self.last_ball is not None:
                print(f"[ball] LOST (had persist={self.ball_persist})", flush=True)
            self.last_ball = None
            self.ball_persist = 0
            self.ball_smooth = None

        # ----- Goal: TTL persistence only -----
        GOAL_TTL = 4
        if fresh_goal is not None:
            self.last_goal = fresh_goal
            self.goal_ttl = GOAL_TTL
            # Memory: remember which side the goal was on, so CARRYING can bias rotation.
            self.last_goal_seen_t = time.time()
            self.last_goal_side  = -1 if fresh_goal['cx'] < (frame.shape[1] / 2.0) else 1
        elif self.goal_ttl > 0:
            self.goal_ttl -= 1
        else:
            self.last_goal = None

        # Reject diagnostic: if model saw raw ball detections but ALL got filtered, log the reasons.
        # Throttled to every 3 seconds so we don't spam.
        if (fresh_ball is None
            and getattr(self, 'last_ball_raw', 0) > 0
            and time.time() - self.last_reject_log_t > 3.0):
            rj = self.last_ball_reject
            print(f"[ball-reject] raw={self.last_ball_raw} reasons: {rj}", flush=True)
            self.last_reject_log_t = time.time()

        ball = self.last_ball
        goal = self.last_goal
        # Anti-flicker: only "act" on ball after persistence threshold reached.
        # TIERED: far balls (small radius) require BALL_PERSIST_FAR_FRAMES, close balls require CHASE_PERSIST_FRAMES.
        if ball is not None:
            needed = BALL_PERSIST_FAR_FRAMES if ball['r'] < BALL_RADIUS_FAR_THRESHOLD else CHASE_PERSIST_FRAMES
            ball_actionable = self.ball_persist >= needed
        else:
            ball_actionable = False

        # ============ MISSION STATE MACHINE ============
        # Handle non-HUNTING states first; fall through to existing chase logic only in HUNTING.
        handled = False

        if self.mission == "CARRYING":
            # Ball is in the front scoop. Just push forward in a straight line.
            # Bail out if ball not seen for too long (false positive triggered CARRYING).
            elapsed = time.time() - self.scan_start
            # Escape detection: ball clearly visible but small (escaped the scoop).
            # 50 px radius ~= 19 cm — definitely not in scoop. Bail to HUNTING.
            ball_escaped = ball_actionable and ball['r'] < 50 and elapsed > 1.0
            if goal:
                self.mission = "ALIGNED"
            elif ball_escaped:
                print(f"[mission] CARRYING bail: ball escaped scoop (r={ball['r']:.0f}, t={elapsed:.1f}s)", flush=True)
                self.mission = "HUNTING"
                stop_motors()
            elif self.ball_ttl == 0 and elapsed > CARRYING_LOST_BAIL_S:
                # Ball was lost — probably false positive. Bail back to HUNTING.
                self.mission = "HUNTING"
                stop_motors()
            elif elapsed > GOAL_SCAN_TIMEOUT:
                self.mission = "HUNTING"
                stop_motors()
            else:
                # Lock camera straight ahead, just push forward
                self.tilt = float(FIXED_TILT)
                self.pan  = 1500.0
                servo_write([[1, int(self.tilt)], [2, int(self.pan)]], 0.1)
                # If we saw the goal recently, bias rotation toward last-known side while pushing.
                # This avoids the full 4-quadrant scan when we already know roughly where the goal is.
                t_since_goal = time.time() - self.last_goal_seen_t
                if t_since_goal < 8.0 and self.last_goal_side != 0:
                    ang = 0.18 * self.last_goal_side    # gentle bias, ball stays in scoop
                    drive(CARRY_FWD * 0.7, ang)
                    cv2.putText(out, f"CARRYING t={elapsed:.1f}s GMEM={'R' if self.last_goal_side > 0 else 'L'} ({t_since_goal:.1f}s ago)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    drive(CARRY_FWD, 0.0)               # straight forward, no rotation
                    cv2.putText(out, f"CARRYING t={elapsed:.1f}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            handled = True

        elif self.mission == "ALIGNED":
            if goal:
                gx_off = goal['cx'] - fx
                if abs(gx_off) < GOAL_CENTERED_PX:
                    self.mission = "SCORING"
                    self.score_start = time.time()
                else:
                    # Rotate while keeping gentle forward pressure (ball stays in scoop)
                    ang = float(np.clip(gx_off * 0.005, -0.4, 0.4))
                    drive(CARRY_FWD * 0.5, ang)
                cv2.putText(out, f"ALIGN_GOAL ex={int(goal['cx']-fx):+d}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            else:
                # Lost goal during align — fall back to scan
                self.mission = "CARRYING"
                self.scan_start = time.time()
            handled = True

        elif self.mission == "SCORING":
            elapsed = time.time() - self.score_start
            if elapsed < PUSH_DURATION_S:
                drive(1.0, 0.0)
                cv2.putText(out, f"SCORING t={elapsed:.1f}/{PUSH_DURATION_S:.1f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                stop_motors()
                self.mission = "HUNTING"
                self.lost = 0
            handled = True

        if handled:
            # Render goal box during mission states so user can see detection
            if goal:
                gx1, gy1, gx2, gy2 = goal['box']
                cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                cv2.putText(out, f"GOAL {goal['conf']:.2f}",
                            (gx1, max(gy1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)
            cv2.line(out, (fx-25, fy), (fx+25, fy), (0,0,255), 2)
            cv2.line(out, (fx, fy-25), (fx, fy+25), (0,0,255), 2)
            cv2.putText(out, f"pan={int(self.pan)} tilt={int(self.tilt)} st={self.mission}",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            with frame_lock:
                latest_frame = out
            return

        if ball_actionable:
            self.lost = 0
            cx, cy, r = ball['cx'], ball['cy'], ball['r']
            ex = cx - fx
            ey = cy - fy

            DEAD = 12
            MAX_STEP = 25           # was 50 — capped lower for smoother camera motion
            ERR_CLAMP = 160         # cap pixel error fed into PID (lost-and-reacquire transients)
            ex_c = float(np.clip(ex, -ERR_CLAMP, ERR_CLAMP))
            ey_c = float(np.clip(ey, -ERR_CLAMP, ERR_CLAMP))
            if abs(ex) > DEAD:
                self.pan_i = float(np.clip(self.pan_i + ex_c, -500, 500))
                d = ex_c - self.pan_last; self.pan_last = ex_c
                step = self.Kp*ex_c + self.Ki*self.pan_i + self.Kd*d
                step = float(np.clip(step, -MAX_STEP, MAX_STEP))
                self.pan -= step
            else:
                self.pan_i = 0; self.pan_last = 0
            if abs(ey) > DEAD:
                self.tilt_i = float(np.clip(self.tilt_i + ey_c, -500, 500))
                d = ey_c - self.tilt_last; self.tilt_last = ey_c
                step = (self.Kp*ey_c + self.Ki*self.tilt_i + self.Kd*d) * 0.7
                step = float(np.clip(step, -MAX_STEP, MAX_STEP))
                self.tilt += step
            else:
                self.tilt_i = 0; self.tilt_last = 0

            self.pan  = float(np.clip(self.pan,  500, 2500))
            self.tilt = float(np.clip(self.tilt, 1700, 2500))   # don't let camera look up at ceiling
            servo_write([[1, int(self.tilt)], [2, int(self.pan)]], 0.18)

            # ===== DECOUPLED BODY STEERING =====
            # Body rotation derives from BALL position in frame (ex), NOT camera servo position.
            # Coupling body to camera caused a feedback oscillation: camera PID swings → body spins →
            # ball moves in frame → camera over-corrects → body spins back → bot wobbles around ball.
            # Now: camera tracks ball independently, body steers directly toward ball center.
            BODY_DEADBAND_PX = 80
            if abs(ex) < BODY_DEADBAND_PX:
                ang = 0.0
            else:
                # Sign matches original: cam_off had opposite sign from ex (since pan -= step), so we negate.
                ang = float(np.clip(-ex * 0.0010, -0.25, 0.25))   # gentle, no overshoot

            d_cm = distance_cm(r)
            # Mission trigger: ball is now in the front scoop -> CARRYING
            # Only trigger if ball has been detected for several consecutive frames
            # (avoids false-positive flicker triggering the state).
            if (d_cm <= CARRY_DIST_CM
                and self.mission == "HUNTING"
                and self.ball_persist >= BALL_PERSIST_FRAMES):
                self.mission = "CARRYING"
                self.scan_start = time.time()
                self.scan_dir = 1 if (cx < fx) else -1
                lin = CARRY_FWD; st = "CAPTURED"
            elif d_cm <= 12:
                lin = 0.85; st = "TOUCH"     # slowed to avoid overshoot before capture
            elif d_cm <= 25:
                lin = 0.95; st = "CLOSE"
            elif d_cm <= 45:
                lin = 1.0;  st = "APPROACH"
            else:
                lin = 1.0;  st = "CHASING"

            # Only slow forward if ball is REALLY off-center, and only modestly.
            if abs(ex) > 150:
                lin *= 0.7

            drive(lin, ang)

            cv2.circle(out, (int(cx), int(cy)), int(r), ball['col'], 2)
            cv2.circle(out, (int(cx), int(cy)), 4, ball['col'], -1)
            cv2.line(out, (fx, fy), (int(cx), int(cy)), (255,255,0), 1)
            cv2.putText(out, f"{ball['name']} {st} d={d_cm:.0f}cm {ball['conf']:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball['col'], 2)
            cv2.putText(out, f"r={int(r)} err=({int(ex)},{int(ey)})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball['col'], 1)
        else:
            self.lost += 1
            self.pan_i = 0; self.tilt_i = 0
            self.pan_last = 0; self.tilt_last = 0
            if self.lost > 5:
                # 3-layer × 4-quadrant scan:
                # Within a quadrant: layer 0 R→L → step up → layer 1 L→R → step up → layer 2 R→L
                # After all 3 layers: TURN body 90° → next quadrant
                if self.search_sub == 'PAN':
                    self.pan += self.search_dir * 14   # was 35 — too fast (224 px/sec). 14 → ~90 px/sec.
                    hit_edge = False
                    if self.pan >= 2300:
                        self.pan = 2300; hit_edge = True
                    elif self.pan <= 700:
                        self.pan = 700; hit_edge = True

                    if hit_edge:
                        if self.search_layer < 2:
                            self.search_layer += 1
                            self.search_dir *= -1
                            print(f"[search] layer {self.search_layer+1}/3", flush=True)
                        else:
                            self.search_sub = 'TURN'
                            self.turn_start_time = time.time()
                            self.search_layer = 0
                            self.search_dir *= -1
                            print(f"[search] TURN start (~90deg, {SEARCH_TURN_DURATION_S}s @ rate {SEARCH_TURN_RATE})", flush=True)

                    target_tilt = SEARCH_TILT_LEVELS[self.search_layer]
                    self.tilt += (target_tilt - self.tilt) * 0.12   # was 0.30 — smoother layer change
                    servo_write([[1, int(self.tilt)], [2, int(self.pan)]], 0.20)   # was 0.08 — gentler servo
                    stop_motors()
                else:   # 'TURN'
                    elapsed = time.time() - self.turn_start_time
                    if elapsed >= SEARCH_TURN_DURATION_S:
                        self.search_sub = 'PAN'
                        stop_motors()
                        print(f"[search] TURN done (rotated {elapsed:.2f}s)", flush=True)
                    else:
                        # Use dedicated high-duty turn — drive() can't break carpet friction
                        turn_in_place(1)
                    target_tilt = SEARCH_TILT_LEVELS[0]
                    self.tilt += (target_tilt - self.tilt) * 0.12   # smoother
                    servo_write([[1, int(self.tilt)], [2, int(self.pan)]], 0.20)   # gentler
            else:
                stop_motors()

            if self.search_sub == 'PAN':
                arrow = '>' if self.search_dir > 0 else '<'
                label = f"SEARCHING L{self.search_layer+1}/3 pan{arrow}"
            else:
                t_remain = SEARCH_TURN_DURATION_S - (time.time() - self.turn_start_time)
                label = f"TURNING ~90deg ({t_remain:.1f}s)"
            cv2.putText(out, label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # ===== DEBUG: render ball + goalpost detections (no Robot-car) =====
        if DEBUG_RENDER_ALL:
            for d in getattr(self, 'last_all', []):
                if d['cls'] == 0:           # skip Robot-car — not used for tracking
                    continue
                if d['cls'] == 1 and d['conf'] < BALL_CONF:
                    continue                 # skip junk low-conf ball boxes
                if d['cls'] == 2 and d['conf'] < GOAL_CONF:
                    continue                 # skip junk low-conf goalpost boxes
                bx1, by1, bx2, by2 = d['box']
                col = (0, 255, 0) if d['cls'] == 1 else (0, 255, 255)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), col, 1)
                cv2.putText(out, f"{d['name']} {d['conf']:.2f}",
                            (bx1, max(by1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # ===== GOALPOST visualization (no behavior change yet — just analyze) =====
        if goal:
            gx1, gy1, gx2, gy2 = goal['box']
            # yellow box for goalpost
            cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
            cv2.putText(out, f"GOAL {goal['conf']:.2f} ar={goal['ratio']:.2f}",
                        (gx1, max(gy1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2)
            # angular offset of goalpost from frame center (would be used for striker/goalie aim)
            g_ex = goal['cx'] - fx
            cv2.putText(out, f"goal_ex={int(g_ex):+d} w={int(goal['w'])} h={int(goal['h'])}",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Show what the model DID see (helps diagnose "no goalpost ever")
            cc = getattr(self, 'last_cls_count', {}) or {}
            cls_names = {0: 'robot', 1: 'ball', 2: 'goal'}
            summary = ' '.join(f"{cls_names.get(k,k)}:{v}" for k, v in cc.items()) or 'none'
            cv2.putText(out, f"no goalpost (model sees: {summary})",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.line(out, (fx-25, fy), (fx+25, fy), (0,0,255), 2)
        cv2.line(out, (fx, fy-25), (fx, fy+25), (0,0,255), 2)
        cv2.putText(out, f"pan={int(self.pan)} tilt={int(self.tilt)}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        with frame_lock:
            latest_frame = out

# ============ Main ============
_loop = True
def _shutdown(*_):
    global _loop
    print("[shutdown] stopping", flush=True)
    _loop = False
    try: stop_motors()
    except Exception: pass

def read_throttled():
    """Returns (raw_int, human_str). Raw 0x0 = healthy.
    Bit 0 = under-voltage NOW, bit 16 = under-voltage HAS OCCURRED since boot.
    """
    try:
        r = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True, timeout=2)
        out = r.stdout.strip()             # "throttled=0x0"
        val = int(out.split('=')[1], 16)
        flags = []
        if val & 0x1:     flags.append("UNDERVOLT-NOW")
        if val & 0x2:     flags.append("FREQ-CAPPED-NOW")
        if val & 0x4:     flags.append("THROTTLED-NOW")
        if val & 0x8:     flags.append("TEMP-LIMIT-NOW")
        if val & 0x10000: flags.append("undervolt-since-boot")
        if val & 0x20000: flags.append("freq-capped-since-boot")
        if val & 0x40000: flags.append("throttled-since-boot")
        if val & 0x80000: flags.append("temp-limit-since-boot")
        return val, (",".join(flags) if flags else "healthy")
    except Exception as e:
        return None, f"vcgencmd-failed:{e}"

def startup_checklist():
    print("=" * 60, flush=True)
    print("[boot] STARTUP CHECKLIST:", flush=True)
    print("[boot]   * battery voltage > 7.4V? (full ~8.4V, brown-out at ~6.5V)", flush=True)
    print("[boot]   * /dev/rrc -> ttyAMA0 (NOT ttyS0)? motors won't talk on ttyS0", flush=True)
    print("[boot]   * carpet vs hardwood? carpet needs >=30% duty to break friction", flush=True)
    val, flags = read_throttled()
    if val is None:
        print(f"[boot]   * throttle status: UNAVAILABLE ({flags})", flush=True)
    elif val == 0:
        print(f"[boot]   * throttle status: healthy (0x0)", flush=True)
    else:
        print(f"[boot]   * throttle status: WARNING 0x{val:x} = {flags}", flush=True)
        print(f"[boot]     ^ previous brown-out detected — battery likely sagged. Check/charge before long runs.", flush=True)
    print("=" * 60, flush=True)

def main():
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    startup_checklist()

    subprocess.run(['fuser', '-k', f'{MJPEG_PORT}/tcp'], capture_output=True)
    time.sleep(0.3)
    start_mjpeg(MJPEG_PORT)
    print(f"[mjpeg] http://192.168.2.2:{MJPEG_PORT}", flush=True)

    cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print(f"[fatal] cannot open {CAMERA_DEV}", flush=True)
        sys.exit(1)

    print(f"[init] loading model: {MODEL_PATH}", flush=True)
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH, task='detect')
    print(f"[init] classes: {model.names}", flush=True)

    tracker = ZhuiYolo(model)

    t_log = time.time(); fps_n = 0
    t_throttle = time.time()
    last_throttle_val = 0
    while _loop:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        fps_n += 1
        tracker.step(frame)
        if time.time() - t_log > 5.0:
            print(f"[loop] fps={fps_n/5.0:.1f}", flush=True)
            fps_n = 0; t_log = time.time()
        # Throttle/brown-out watcher: log when status changes (catches battery sag before reboot).
        if time.time() - t_throttle > 10.0:
            val, flags = read_throttled()
            if val is not None and val != last_throttle_val:
                print(f"[power] throttle 0x{val:x} = {flags}", flush=True)
                if val & 0x1:
                    print(f"[power] !!! UNDER-VOLTAGE NOW — battery sagging, brown-out imminent. Stop motors & charge.", flush=True)
                last_throttle_val = val
            t_throttle = time.time()

    cap.release()
    stop_motors()
    servo_write([[1, 1800], [2, 1500]], 0.5)
    print("[main] done", flush=True)

if __name__ == '__main__':
    main()
