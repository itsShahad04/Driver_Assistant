# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import math
import numpy as np

model = YOLO("best.pt")

img_path = "z.jpg"  

img = cv2.imread(img_path)

if img is None:
    print("Error: Could not read image.")
    print("Make sure the image file exists in the same folder.")
    exit()

results = model(img_path, conf=0.25)
r = results[0]

boxes = r.boxes
empty_spaces = []
occupied_spaces = []
names = model.names

for box in boxes:
    cls_id = int(box.cls[0])
    cls_name = names[cls_id]

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    spot_info = {
        "box": (x1, y1, x2, y2),
        "center": (cx, cy),
        "conf": conf,
        "class": cls_name
    }

    if cls_name == "empty":
        empty_spaces.append(spot_info)
    elif cls_name == "occupied":
        occupied_spaces.append(spot_info)

print("Empty:", len(empty_spaces))
print("Occupied:", len(occupied_spaces))

best_spot = None

if len(empty_spaces) > 0:
    entrance_point = (0, img.shape[0])  

    def distance_to_entrance(spot):
        cx, cy = spot["center"]
        ex, ey = entrance_point
        return math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)

    best_spot = min(empty_spaces, key=distance_to_entrance)

output = img.copy()

for spot in occupied_spaces:
    x1, y1, x2, y2 = spot["box"]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

for spot in empty_spaces:
    x1, y1, x2, y2 = spot["box"]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

if best_spot is not None:
    x1, y1, x2, y2 = best_spot["box"]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 4)
    cv2.putText(output, "Best Spot", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

total_spaces = len(empty_spaces) + len(occupied_spaces)

if len(empty_spaces) > 0:
    status_text = "Parking Available"
else:
    status_text = "Parking Full"

h, w, _ = output.shape

panel_width = 250

new_output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)

new_output[:] = (30, 30, 30)

new_output[:, :w] = output

cv2.putText(new_output, f"Empty: {len(empty_spaces)}", (w + 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.putText(new_output, f"Occupied: {len(occupied_spaces)}", (w + 20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.putText(new_output, f"Total: {total_spaces}", (w + 20, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.putText(new_output, status_text, (w + 20, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

cv2.imshow("Smart Driver Assistant", new_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", new_output)
print("Saved result as result.jpg")