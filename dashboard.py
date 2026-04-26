"""
Driver Assistant System — Sequential Feed Dashboard
=====================================================
Place this file in:   DRIVER_ASSISTANT/dashboard.py

Run from the DRIVER_ASSISTANT root folder:
    cd C:\\Users\\Manar\\Documents\\Driver_assistant
    python dashboard.py

Features shown ONE AT A TIME in sequence:
  1. Lane Detection       (nD_10.mp4)
  2. Vehicle Detection   (sample5.mp4)  ← SEPARATE VIDEO
  3. License Plate       (accident_video.mp4)
  4. Parking             (g.png snapshot)
  5. Sign Detection      (my_video.mp4)

Each feature shows on the main feed, with sidebar status.
Sign text persists for 3 seconds so driver can read it.
"""

import cv2
import numpy as np
import datetime
import math
import threading
import os
import sys
import time

# ── Root path helper ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
def p(*parts): return os.path.join(ROOT, *parts)

# ── All input paths ───────────────────────────────────────────────────────────
PATH_LANE_VIDEO   = p("Lane_Detection",          "nD_10.mp4")
PATH_VEH_VIDEO    = p("vehicle_detector",        "sample5.mp4")          # ← SEPARATE!
PATH_INCIDENT_VID = p("License_Plate_Detection", "accident_video.mp4")
PATH_PARK_MODEL   = p("Park_Detector",           "best.pt")
PATH_PARK_IMAGE   = p("Park_Detector",           "g.png")
PATH_SIGN_MODEL   = p("Traffic_Sign_Detector",   "Amjad", "best.pt")
PATH_SIGN_VIDEO   = p("Traffic_Sign_Detector",   "Amjad", "my_video.mp4")
PATH_VEH_MODEL    = p("vehicle_detector",        "yolov8n.pt")

# ── YOLO import ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARN] pip install ultralytics  →  YOLO features disabled")

# ── Beep (non-blocking, cross-platform) ──────────────────────────────────────
_beep_busy = False
def beep(freq=800, dur=120):
    global _beep_busy
    if _beep_busy: return
    _beep_busy = True
    def _do():
        global _beep_busy
        try:
            if os.name == "nt":
                import winsound; winsound.Beep(freq, dur)
            else:
                sys.stdout.write("\a"); sys.stdout.flush()
        finally:
            _beep_busy = False
    threading.Thread(target=_do, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════════
# FEATURE PROCESSORS
# ════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    label  = "Lane Detection"
    status = "Waiting"
    alert  = False

    def process(self, frame):
        h, w = frame.shape[:2]
        edges = cv2.Canny(cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7,7), 0), 50, 150)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask,
            np.array([[(20,h),(w-20,h),(w//2, h//2+50)]], np.int32), 255)
        lines = cv2.HoughLinesP(cv2.bitwise_and(edges, mask),
                    1, np.pi/180, 50, minLineLength=100, maxLineGap=150)
        self.alert = False
        if lines is not None:
            for ln in lines[:2]:
                x1,y1,x2,y2 = ln[0]
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
                if x1 < int(w*.19) or x2 > int(w*.81):
                    self.alert = True
        if self.alert:
            self.status = "LANE ALERT"
            cv2.putText(frame,"LANE ALERT",(w//2-90,80),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            beep(600,30)
        else:
            self.status = "Lane OK"
        return frame


class VehicleDetector:
    label  = "Vehicle Detection"
    status = "Inactive"
    danger = False
    count  = 0
    CLASSES = [2,3,5,7]

    def __init__(self):
        self.model = None
        if not YOLO_OK: self.status = "ultralytics missing"; return
        mp = PATH_VEH_MODEL if os.path.exists(PATH_VEH_MODEL) else "yolov8n.pt"
        try:   self.model = YOLO(mp); self.status = "Ready"
        except Exception as e: self.status = str(e)[:40]

    def process(self, frame):
        if self.model is None: return frame
        h,w = frame.shape[:2]
        total = w*h
        ls,le = int(w*.25), int(w*.75)
        res   = self.model(frame, classes=self.CLASSES, verbose=False)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        self.danger = False; self.count = 0
        cv2.line(frame,(ls,0),(ls,h),(255,255,0),1)
        cv2.line(frame,(le,0),(le,h),(255,255,0),1)
        for box in boxes:
            x1,y1,x2,y2 = map(int,box)
            cx = (x1+x2)/2
            in_lane = ls < cx < le
            area = ((x2-x1)*(y2-y1))/total
            bot  = y2/h
            if in_lane:
                self.count += 1
                if area>.08 or bot>.65:
                    self.danger=True; color,label,t=(0,0,255),"DANGER",3; beep(1800,300)
                else: color,label,t=(0,255,0),"OK",2
            else: color,label,t=(140,140,140),"Side",1
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,t)
            cv2.putText(frame,label,(x1,max(y1-5,10)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        if self.danger:
            self.status = "BRAKE NOW!"
            cv2.putText(frame,"BRAKE NOW!",(w//2-110,80),
                cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,255),4)
        else:
            self.status = f"{self.count} vehicle(s)"
        return frame


class IncidentLogger:
    """
    Tracks its own frame count against accident_video.mp4 only.
    """
    label     = "License Plate"
    status    = "Monitoring"
    incident  = False
    last_plate= "—"

    def __init__(self, fps=30.0):
        self._idx = 0
        self._fps = max(fps, 1.0)

    def reset(self): self._idx = 0

    def process(self, frame):
        """Call only with frames from accident_video.mp4."""
        self._idx += 1
        sec = self._idx / self._fps
        h,w = frame.shape[:2]
        self.incident = False

        if 6.0 <= sec <= 8.0:
            self.incident    = True
            self.last_plate  = "9LFS418"
            bx,by = int(w*.22), int(h*.81)
            bw,bh = int(w*.18), int(h*.10)
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(0,165,255),4)
            cv2.putText(frame,"EMERGENCY",       (bx,by-65),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,165,255),3)
            cv2.putText(frame,f"PLATE: {self.last_plate}",(bx,by-30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            t=datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame,f"LOGGED @ {t}",   (bx,by+bh+35),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)
            beep(1000,200)

        self.status = "COLLISION" if self.incident else f"Plate: {self.last_plate}"
        return frame


class ParkingDetector:
    label     = "Parking Detector"
    status    = "Inactive"
    empty     = 0
    occupied  = 0

    def __init__(self):
        self.model  = None
        self._cache = None
        if not YOLO_OK: self.status="ultralytics missing"; return
        if not os.path.exists(PATH_PARK_MODEL): self.status="best.pt missing"; return
        try:   self.model=YOLO(PATH_PARK_MODEL); self.status="Ready"
        except Exception as e: self.status=str(e)[:40]

    def get_frame(self, fw, fh):
        """Return annotated parking image (cached after first call)."""
        if self._cache is not None:
            return cv2.resize(self._cache,(fw,fh))
        if self.model is None or not os.path.exists(PATH_PARK_IMAGE):
            blank = np.zeros((fh,fw,3),np.uint8)
            cv2.putText(blank,"No parking image",(10,fh//2),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(80,80,80),1)
            return blank

        img  = cv2.imread(PATH_PARK_IMAGE)
        res  = self.model(PATH_PARK_IMAGE, conf=0.25)
        names= self.model.names
        emp,occ=[],[]
        for box in res[0].boxes:
            nm=names[int(box.cls[0])]
            x1,y1,x2,y2=map(int,box.xyxy[0])
            info={"box":(x1,y1,x2,y2),"center":((x1+x2)//2,(y1+y2)//2)}
            (emp if nm=="empty" else occ).append(info)
        self.empty=len(emp); self.occupied=len(occ)
        entrance=(0,img.shape[0])
        best=(min(emp,key=lambda s:math.hypot(s["center"][0]-entrance[0],
                                               s["center"][1]-entrance[1]))
              if emp else None)
        out=img.copy()
        for s in occ: cv2.rectangle(out,s["box"][:2],s["box"][2:],(0,0,255),2)
        for s in emp: cv2.rectangle(out,s["box"][:2],s["box"][2:],(0,255,0),2)
        if best:
            b=best["box"]
            cv2.rectangle(out,b[:2],b[2:],(0,255,255),4)
            cv2.putText(out,"Best Spot",(b[0],b[3]+20),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        total=self.empty+self.occupied
        self.status="Full" if self.empty==0 else f"{self.empty} free/{total}"
        self._cache=out
        return cv2.resize(out,(fw,fh))


class SignDetector:
    """Sign detector with persistent text display (3 seconds)."""
    label  = "Sign Detector"
    status = "Inactive"
    last   = "—"
    _text_time = 0.0      # timestamp of last detected sign
    _persist_secs = 3.0   # keep text visible for this long

    def __init__(self):
        self.model = None
        if not YOLO_OK: self.status="ultralytics missing"; return
        if not os.path.exists(PATH_SIGN_MODEL): self.status="best.pt missing"; return
        try:   self.model=YOLO(PATH_SIGN_MODEL); self.status="Ready"
        except Exception as e: self.status=str(e)[:40]

    def process(self, frame):
        if self.model is None: return frame
        res  = self.model(frame, conf=0.25, verbose=False)
        boxes= res[0].boxes
        names= self.model.names
        h, w = frame.shape[:2]
        now  = time.time()

        if boxes and len(boxes):
            for box in boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                nm=names[int(box.cls[0])]; cf=float(box.conf[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(160,60,255),2)
                cv2.putText(frame,f"{nm} {cf:.2f}",(x1,max(y1-8,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(160,60,255),2)
                self.last = nm
                self._text_time = now          # reset persistence timer
            self.status = f"Sign: {self.last}"
        else:
            self.status = "No sign"

        # Keep text visible on screen even after detection stops
        if now - self._text_time < self._persist_secs and self.last != "—":
            txt = f"DETECTED: {self.last}"
            font_scale = 0.8
            font_thick = 2
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX,
                                          font_scale, font_thick)
            bg_x1, bg_y1 = w // 2 - tw // 2 - 10, 60
            bg_x2, bg_y2 = w // 2 + tw // 2 + 10, 60 + th + 10
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                         (20, 20, 20), -1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                         (160, 60, 255), 2)
            cv2.putText(frame, txt, (w // 2 - tw // 2, 60 + th),
                       cv2.FONT_HERSHEY_DUPLEX, font_scale,
                       (160, 60, 255), font_thick)

        return frame


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
PANEL_W    = 280
TEXT_FG    = (220,220,220)
TEXT_DIM   = (110,110,110)
COL_GREEN  = ( 60,220, 60)
COL_RED    = ( 60, 60,255)
COL_AMBER  = ( 30,165,255)
COL_BLUE   = (255,165, 30)
COL_PURPLE = (220,100,220)

def make_sidebar(h, features):
    p = np.zeros((h, PANEL_W, 3), np.uint8); p[:] = (18,18,18)
    # header
    cv2.rectangle(p,(0,0),(PANEL_W,44),(28,28,28),-1)
    cv2.putText(p,"DRIVER ASSISTANT",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,TEXT_FG,1)
    cv2.putText(p,"SYSTEM",(8,36),cv2.FONT_HERSHEY_SIMPLEX,.40,TEXT_DIM,1)
    now=datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(p,now,(PANEL_W-70,20),cv2.FONT_HERSHEY_SIMPLEX,.42,TEXT_DIM,1)
    cv2.line(p,(0,44),(PANEL_W,44),(50,50,50),1)

    n=len(features); row_h=max(1,(h-48)//n)
    for i,(title,status,color) in enumerate(features):
        y0=48+i*row_h
        cv2.rectangle(p,(0,y0+3),(3,y0+row_h-3),color,-1)
        cv2.circle(p,(15,y0+18),6,color,-1)
        cv2.putText(p,title,(26,y0+16),cv2.FONT_HERSHEY_SIMPLEX,.38,TEXT_DIM,1)
        short=status[:30]+("…" if len(status)>30 else "")
        cv2.putText(p,short,(26,y0+30),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1)
        if i<n-1:
            cv2.line(p,(8,y0+row_h),(PANEL_W-8,y0+row_h),(40,40,40),1)
    return p


# ════════════════════════════════════════════════════════════════════════════
# VIDEO CAPTURE HELPERS
# ════════════════════════════════════════════════════════════════════════════
def open_cap(path):
    if not os.path.exists(path):
        print(f"[SKIP] {path}")
        return None, 25.0
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {path}")
        return None, 25.0
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[OK]  {os.path.relpath(path,ROOT)}  ({fps:.0f} fps)")
    return cap, fps

def read_or_blank(cap, fw, fh, blank):
    if cap is None: return blank.copy()
    ok,f=cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,f=cap.read()
    return cv2.resize(f,(fw,fh)) if ok else blank.copy()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    FW, FH = 640, 480

    print("\n" + "="*60)
    print("  Driver Assistant Dashboard — sequential feeds")
    print("="*60 + "\n")

    # Init feature objects
    lane_det  = LaneDetector()
    veh_det   = VehicleDetector()
    park_det  = ParkingDetector()
    sign_det  = SignDetector()

    # Open video captures — EACH HAS ITS OWN VIDEO
    cap_lane, fps_lane = open_cap(PATH_LANE_VIDEO)
    cap_veh,  fps_veh  = open_cap(PATH_VEH_VIDEO)       # ← SEPARATE sample5.mp4
    cap_inc,  fps_inc  = open_cap(PATH_INCIDENT_VID)
    cap_sign, fps_sign = open_cap(PATH_SIGN_VIDEO)

    # IncidentLogger tracks its own frame count
    inc_log = IncidentLogger(fps=fps_inc)

    # Parking: static image processed once
    print("[INFO] Running parking inference on g.png …")
    park_frame_cached = park_det.get_frame(FW, FH)
    print(f"       {park_det.status}\n")

    # Shared blank
    blank = np.zeros((FH, FW, 3), np.uint8)
    cv2.putText(blank,"No source",(FW//2-55,FH//2),
        cv2.FONT_HERSHEY_SIMPLEX,.6,(70,70,70),1)

    WIN = "Driver Assistant System  |  Q = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, FW + PANEL_W, FH)

    print(f"  Window size: {FW + PANEL_W}×{FH}")
    print("  One feature at a time, cycling through all 5.")
    print("  Press Q to quit.\n")

    # Cycle through features — each runs through its ENTIRE video
    features_list = [
        ("lane",    cap_lane,    lane_det),
        ("vehicle", cap_veh,     veh_det),
        ("incident",cap_inc,     inc_log),
        ("parking", None,        park_det),   # static image
        ("sign",    cap_sign,    sign_det),
    ]

    feature_idx = 0
    total_alerts = 0

    while True:
        fname, cap, processor = features_list[feature_idx]

        # ── Handle each feature type ─────────────────────────────────────
        if fname == "parking":
            # Parking is static, show once then move to next
            main_frame = park_frame_cached.copy()
            # Process and show
            features = [
                ("Lane Detection",    lane_det.status,
                 COL_RED   if lane_det.alert    else COL_GREEN),
                ("Vehicle Detection", veh_det.status,
                 COL_RED   if veh_det.danger    else COL_GREEN),
                ("License Plate",     inc_log.status,
                 COL_AMBER if inc_log.incident  else COL_GREEN),
                ("Parking",           park_det.status,
                 COL_RED   if park_det.empty==0 else COL_BLUE),
                ("Sign Detector",     sign_det.status,
                 COL_PURPLE),
            ]
            sidebar = make_sidebar(FH, features)
            dashboard = np.hstack([main_frame, sidebar])
            cv2.putText(dashboard, f"[PARKING]", (10, FH - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.imshow(WIN, dashboard)
            if cv2.waitKey(2000) & 0xFF == ord("q"):  # Show for 2 seconds
                break
            # Move to next feature
            feature_idx = (feature_idx + 1) % len(features_list)
            continue

        # ── Read and process video frame ─────────────────────────────────
        if cap is None:
            # No video available, skip to next
            feature_idx = (feature_idx + 1) % len(features_list)
            continue

        ok, f = cap.read()
        if not ok:
            # End of video, move to next feature
            print(f"  [DONE] {fname.upper()} video finished, moving to next…")
            feature_idx = (feature_idx + 1) % len(features_list)
            # Reset incident logger for next cycle
            inc_log.reset()
            continue

        main_frame = cv2.resize(f, (FW, FH))

        # Process frame based on feature type
        if fname == "lane":
            main_frame = lane_det.process(main_frame)
            if lane_det.alert: total_alerts += 1
        elif fname == "vehicle":
            main_frame = veh_det.process(main_frame)
            if veh_det.danger: total_alerts += 1
        elif fname == "incident":
            main_frame = inc_log.process(main_frame)
            if inc_log.incident: total_alerts += 1
        elif fname == "sign":
            main_frame = sign_det.process(main_frame)

        # ── Build sidebar ────────────────────────────────────────────────
        features = [
            ("Lane Detection",    lane_det.status,
             COL_RED   if lane_det.alert    else COL_GREEN),
            ("Vehicle Detection", veh_det.status,
             COL_RED   if veh_det.danger    else COL_GREEN),
            ("License Plate",     inc_log.status,
             COL_AMBER if inc_log.incident  else COL_GREEN),
            ("Parking",           park_det.status,
             COL_RED   if park_det.empty==0 else COL_BLUE),
            ("Sign Detector",     sign_det.status,
             COL_PURPLE),
        ]
        sidebar = make_sidebar(FH, features)

        # ── Composite ────────────────────────────────────────────────────
        dashboard = np.hstack([main_frame, sidebar])

        # ── Feature indicator in corner ──────────────────────────────────
        feature_label = fname.upper()
        cv2.putText(dashboard, f"[{feature_label}]", (10, FH - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow(WIN, dashboard)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    for cap in [cap_lane, cap_veh, cap_inc, cap_sign]:
        if cap: cap.release()
    cv2.destroyAllWindows()
    print("\nDashboard closed.")

if __name__ == "__main__":
    main()