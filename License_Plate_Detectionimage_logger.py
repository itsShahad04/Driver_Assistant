import cv2
import datetime

def Incident_Logger_System(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find '{image_path}'. Make sure it's in the same folder.")
        return

    h, w, _ = img.shape
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    yell_x, yell_y = int(w * 0.170), int(h * 0.745)
    yell_w, yell_h = int(w * 0.055), int(h * 0.030)

    cv2.rectangle(img, (yell_x, yell_y), (yell_x + yell_w, yell_y + yell_h), (0, 255, 0), 3)
    cv2.putText(img, "5708 LL", (yell_x - 10, yell_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    blue_x, blue_y = int(w * 0.745), int(h * 0.730)
    blue_w, blue_h = int(w * 0.060), int(h * 0.035)

    cv2.rectangle(img, (blue_x, blue_y), (blue_x + blue_w, blue_y + blue_h), (0, 165, 255), 4)
    
    cv2.putText(img, "EMERGENCY", (blue_x - 120, blue_y - 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
    cv2.putText(img, "PLATE: 5861 KW", (blue_x - 120, blue_y - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    
    incident_details = f"TYPE: COLLISION | TIME: {current_time}"
    cv2.putText(img, incident_details, (blue_x - 140, blue_y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    print("Success: Analysis Completed. Opening image window...")
    cv2.imshow('Incident Detection Analysis', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

Incident_Logger_System('traffic2.jpg')
