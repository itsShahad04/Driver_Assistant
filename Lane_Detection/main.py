import cv2
import numpy as np
import winsound

video_path = 'nD_10.mp4' 
cap = cv2.VideoCapture(video_path)


frame_width = 640
frame_height = 480

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Final_Project_Result.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    mask = np.zeros_like(edges)
    polygon = np.array([[(20, frame_height), (frame_width-20, frame_height), (frame_width//2, frame_height//2 + 50)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=150)
    
    if lines is not None:
        for line in lines[:2]:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            if x1 < 120 or x2 > 520:
                winsound.Beep(600, 30) 
                cv2.putText(frame, "LANE ALERT", (220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    cv2.imshow('Recording Result...', frame)
    
   
    out.write(frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'): break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Final_Project_Result.avi")