import cv2
import numpy as np

# Import your teammates' modules
import lane_detection as ld
import sign_recognition as sr
import plate_detector as pd
import parking_detector as pk

def create_dashboard(lane_img, sign_img, plate_img, parking_img):
    """Combines four images into a 2x2 grid with a status panel."""
    
    # 1. Resize all outputs to a standard size (e.g., 640x360)
    size = (640, 360)
    img1 = cv2.resize(lane_img, size)
    img2 = cv2.resize(sign_img, size)
    img3 = cv2.resize(plate_img, size)
    img4 = cv2.resize(parking_img, size)

    # 2. Create the Grid
    top_row = np.hstack((img1, img2))
    bottom_row = np.hstack((img3, img4))
    grid = np.vstack((top_row, bottom_row))

    # 3. Create the Side Info Panel (Black background)
    panel_w = 300
    panel = np.zeros((grid.shape[0], panel_w, 3), dtype=np.uint8)
    
    # Add text to the panel (UX design)
    cv2.putText(panel, "SYSTEM STATUS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, "Lanes: ACTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(panel, "Parking: FULL", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # 4. Final Stitch
    final_output = np.hstack((grid, panel))
    return final_output

def main():
    # Use a generic video or webcam as the source
    cap = cv2.VideoCapture('traffic_test.mp4')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Run all features simultaneously on the same frame
        lane_out = ld.process_lane(frame.copy())
        sign_out = sr.process_signs(frame.copy())
        plate_out = pd.process_plates(frame.copy())
        parking_out = pk.process_parking(frame.copy())

        # Combine into dashboard
        dashboard = create_dashboard(lane_out, sign_out, plate_out, parking_out)

        cv2.imshow("Integrated AI Driver Assistance System", dashboard)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()