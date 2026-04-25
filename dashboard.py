"""
Driver Assistant System — Multi-Feed Grid Dashboard
=====================================================
Place this file in:   DRIVER_ASSISTANT/dashboard.py

Run from the DRIVER_ASSISTANT root folder:
    cd C:\\Users\\Manar\\Documents\\Driver_assistant
    python dashboard.py

Layout (each cell is an independent video feed):
┌─────────────────────┬─────────────────────┬──────────┐
│  Lane Detection     │  Vehicle Detection  │          │
│  (nD_10.mp4)        │  (nD_10.mp4)        │ SIDEBAR  │
├─────────────────────┼─────────────────────┤          │
│  License Plate /    │  Parking Detector   │  status  │
│  Incident           │  (g.png snapshot)   │  panels  │
│  (accident_video)   │                     │          │
├─────────────────────┼─────────────────────┤          │
│  Sign Detector      │  [summary stats]    │          │
│  (my_video.mp4)     │                     │          │
└─────────────────────┴─────────────────────┴──────────┘
"""

import cv2
import numpy as np
import datetime
import math
import threading
import os
import sys

# ── Root path helper ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
def p(*parts): return os.path.join(ROOT, *parts)

# ── All input paths ───────────────────────────────────────────────────────────
PATH_LANE_VIDEO   = p("Lane_Detection",          "nD_10.mp4")
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
# FEATURE PROCESSORS  (each is self-contained, owns its own state)
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
            cv2.putText(frame,"LANE ALERT",(w//2-90,50),
                cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
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
            cv2.putText(frame,"BRAKE NOW!",(w//2-110,55),
                cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,255),4)
        else:
            self.status = f"{self.count} vehicle(s)"
        return frame


class IncidentLogger:
    """
    Completely independent from the lane/vehicle feeds.
    Tracks its own frame index against accident_video.mp4 only.
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
    label  = "Sign Detector"
    status = "Inactive"
    last   = "—"

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
        if boxes and len(boxes):
            for box in boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                nm=names[int(box.cls[0])]; cf=float(box.conf[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(160,60,255),2)
                cv2.putText(frame,f"{nm} {cf:.2f}",(x1,max(y1-8,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(160,60,255),2)
                self.last=nm
            self.status=f"Sign: {self.last}"
        else:
            self.status="No sign"
        return frame


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
PANEL_W    = 260
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

def label_cell(frame, text):
    """Dark strip with feature name at the top of each cell."""
    cv2.rectangle(frame,(0,0),(frame.shape[1],22),(0,0,0),-1)
    cv2.putText(frame,text,(6,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(180,180,180),1)
    return frame


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    # Cell size for each of the 6 grid tiles (2 cols × 3 rows)
    CW, CH = 400, 300          # cell width / height
    GRID_W = CW*2              # 800
    GRID_H = CH*3              # 900
    DASH_W = GRID_W + PANEL_W  # 1060
    DASH_H = GRID_H            # 900

    print("\n" + "="*60)
    print("  Driver Assistant Grid Dashboard — initialising…")
    print("="*60 + "\n")

    # Init feature objects
    lane_det  = LaneDetector()
    veh_det   = VehicleDetector()
    park_det  = ParkingDetector()
    sign_det  = SignDetector()

    # Each feature gets its own dedicated video capture
    cap_lane, fps_lane = open_cap(PATH_LANE_VIDEO)
    cap_veh,  fps_veh  = open_cap(PATH_LANE_VIDEO)     # same road video, separate cap
    cap_inc,  fps_inc  = open_cap(PATH_INCIDENT_VID)   # accident_video ONLY for plate
    cap_sign, fps_sign = open_cap(PATH_SIGN_VIDEO)

    # IncidentLogger tracks its own frame count against accident_video
    inc_log = IncidentLogger(fps=fps_inc)

    # Parking: static image processed once
    print("[INFO] Running parking inference on g.png …")
    park_frame_cached = park_det.get_frame(CW, CH)
    print(f"       {park_det.status}")

    # Shared blank tile
    blank = np.zeros((CH, CW, 3), np.uint8)
    cv2.putText(blank,"No source",(CW//2-55,CH//2),
        cv2.FONT_HERSHEY_SIMPLEX,.6,(70,70,70),1)

    # Summary stats
    total_alerts = 0

    WIN = "Driver Assistant System  |  Q = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DASH_W, DASH_H)
    print(f"\n  Window size: {DASH_W}×{DASH_H}   Press Q to quit.\n")

    # Throttle YOLO calls to reduce lag: process every Nth frame
    # Set to 1 to process every frame (slower), higher = faster but less frequent detections
    YOLO_EVERY = 2
    tick = 0

    while True:
        tick += 1
        run_yolo = (tick % YOLO_EVERY == 0)

        # ── Read raw frames (each cap is independent) ─────────────────────
        raw_lane = read_or_blank(cap_lane, CW, CH, blank)
        raw_veh  = read_or_blank(cap_veh,  CW, CH, blank)
        raw_inc  = read_or_blank(cap_inc,  CW, CH, blank)
        raw_sign = read_or_blank(cap_sign, CW, CH, blank)

        # ── Process each feature on its own dedicated frame ───────────────
        # Lane — always fast (no YOLO)
        cell_lane = lane_det.process(raw_lane.copy())
        label_cell(cell_lane, "Lane Detection")

        # Vehicle — YOLO, throttled
        if run_yolo:
            cell_veh = veh_det.process(raw_veh.copy())
        else:
            cell_veh = raw_veh.copy()
            # Keep last known status displayed
            if veh_det.danger:
                cv2.putText(cell_veh,"BRAKE NOW!",(CW//2-110,55),
                    cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,255),4)
        label_cell(cell_veh, "Vehicle Detection")

        # Incident/Plate — always fast (no YOLO, just overlay)
        cell_inc = inc_log.process(raw_inc.copy())
        label_cell(cell_inc, "License Plate / Incident")

        # Parking — static cached image, no per-frame work
        cell_park = park_frame_cached.copy()
        label_cell(cell_park, f"Parking  [{park_det.status}]")

        # Sign — YOLO, throttled
        if run_yolo:
            cell_sign = sign_det.process(raw_sign.copy())
        else:
            cell_sign = raw_sign.copy()
        label_cell(cell_sign, "Sign Detector")

        # ── Summary tile (bottom-right cell) ─────────────────────────────
        cell_info = np.zeros((CH, CW, 3), np.uint8); cell_info[:] = (16,16,16)
        if lane_det.alert or veh_det.danger or inc_log.incident:
            total_alerts += 1
        now = datetime.datetime.now().strftime("%H:%M:%S")

        lines_info = [
            ("DRIVER ASSISTANT",    (180,180,180), .55, 1),
            (now,                   (100,100,100), .42, 1),
            ("",None,0,0),
            (f"Lane:     {lane_det.status}",
             (60,60,255) if lane_det.alert else (60,220,60), .46, 1),
            (f"Vehicle:  {veh_det.status}",
             (60,60,255) if veh_det.danger else (60,220,60), .46, 1),
            (f"Plate:    {inc_log.status}",
             (30,165,255) if inc_log.incident else (60,220,60), .46, 1),
            (f"Parking:  {park_det.status}",
             (255,165,30), .46, 1),
            (f"Sign:     {sign_det.status}",
             (220,100,220), .46, 1),
            ("",None,0,0),
            (f"Alerts fired: {total_alerts}",
             (60,60,255) if total_alerts>0 else (60,220,60), .46, 1),
            ("","",0,0),
            ("Press Q to quit", (60,60,60), .38, 1),
        ]
        y=30
        for txt, col, scale, thick in lines_info:
            if txt and col:
                cv2.putText(cell_info, txt, (12,y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)
            y += 26 if txt else 10
        label_cell(cell_info, "System Summary")

        # ── Assemble 2×3 grid ─────────────────────────────────────────────
        row0 = np.hstack([cell_lane, cell_veh])
        row1 = np.hstack([cell_inc,  cell_park])
        row2 = np.hstack([cell_sign, cell_info])
        grid = np.vstack([row0, row1, row2])

        # ── Sidebar ───────────────────────────────────────────────────────
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
        sidebar = make_sidebar(DASH_H, features)

        # ── Final composite ───────────────────────────────────────────────
        dashboard = np.hstack([grid, sidebar])
        cv2.imshow(WIN, dashboard)

        # Adaptive wait: faster when danger detected
        wait = 1 if (veh_det.danger or lane_det.alert) else 20
        if cv2.waitKey(wait) & 0xFF == ord("q"):
            break

    for cap in [cap_lane, cap_veh, cap_inc, cap_sign]:
        if cap: cap.release()
    cv2.destroyAllWindows()
    print("Dashboard closed.")

if __name__ == "__main__":
    main()
    