import cv2
import datetime

def Incident_Logger_System_VSCode(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not find the video file. Check the file path.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    output_name = 'AccidentVideo_Result.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_count += 1
        current_second = frame_count / fps
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        if 6.0 <= current_second <= 8.0: 
            x, y = int(width * 0.22), int(height * 0.81) 
            w, h = int(width * 0.18), int(height * 0.10)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 4)
            
            cv2.putText(frame, "EMERGENCY", (x, y - 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
            cv2.putText(frame, "PLATE: 9LFS418", (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"LOGGED @ {current_time}", (x, y + h + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

        cv2.imshow('Wafa Project - Video Analysis', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Process Completed! Result saved as: {output_name}")

Incident_Logger_System_VSCode('accident_video.mp4')