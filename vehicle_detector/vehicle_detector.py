import cv2
import threading
import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'yolov8n.pt')

# Reloading the model with the corrected path
try:
    model = YOLO(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

is_beeping = False

def play_beep():
    global is_beeping
    if os.name == 'nt': 
        import winsound
        winsound.Beep(1800, 300) # Sharp alert sound
    else: 
        print('\a') 
    is_beeping = False

def trigger_alert():
    global is_beeping
    if not is_beeping:
        is_beeping = True
        threading.Thread(target=play_beep).start()

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_area = width * height

    # --- LANE FOCUS SETTINGS ---
    # We focus on the middle 50% of the screen (from 25% to 75%)
    LANE_START = int(width * 0.25)
    LANE_END = int(width * 0.75)
    
    # --- SENSITIVITY FOR CENTER CARS ---
    # Since front cars are smaller for longer, we use a more sensitive area limit
    CENTER_AREA_LIMIT = 0.08  # Alert if car takes up 8% of the screen in your lane
    CENTER_BOTTOM_LIMIT = 0.65 # Alert if car bottom is 65% down the screen

    win_name = "Lane-Focused Driver Assist"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        danger_in_lane = False
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Visual guide for lane (Optional: draws two lines on the screen)
        cv2.line(frame, (LANE_START, 0), (LANE_START, height), (255, 255, 0), 1)
        cv2.line(frame, (LANE_END, 0), (LANE_END, height), (255, 255, 0), 1)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) / 2
            
            # CHECK: Is this car in our lane?
            is_in_our_lane = LANE_START < center_x < LANE_END
            
            # Metrics
            area_pct = ((x2 - x1) * (y2 - y1)) / total_area
            bottom_pct = y2 / height

            if is_in_our_lane:
                # If it's in our lane and meets thresholds -> RED
                if area_pct > CENTER_AREA_LIMIT or bottom_pct > CENTER_BOTTOM_LIMIT:
                    danger_in_lane = True
                    color, label, thick = (0, 0, 255), "FRONT DANGER", 3
                else:
                    color, label, thick = (0, 255, 0), "Following", 2
            else:
                # If it's on the sides -> GHOST/THIN boxes (Ignore these for alerts)
                color, label, thick = (150, 150, 150), "Side Lane", 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, f"A:{area_pct:.2f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if danger_in_lane:
            trigger_alert()
            cv2.putText(frame, "BRAKE NOW!", (width//2 - 100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
def main():
    model = YOLO('yolov8n.pt')
    video_files = ['sample2.mp4', 'sample5.mp4']
    
    for v in video_files:
        process_video(v, model)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()