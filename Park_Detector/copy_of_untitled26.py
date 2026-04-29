# -*- coding: utf-8 -*-
#I imported YOLO for detection, OpenCV for image processing, math to calculate distance, and NumPy to create the side panel.
from ultralytics import YOLO
import cv2
import math
import numpy as np
#Here I load my trained model, then I read the input image.
model = YOLO("best.pt")
img_path = "z.jpg" 
#هنا عشان افتح الصوره 
img = cv2.imread(img_path)
#This checks if the image was loaded correctly. If not, the program stops.
if img is None:
    print("Error: Could not read image.")
    print("Make sure the image file exists in the same folder.")
    exit()
#يعرض نتايج يلي واثق منها The model processes the image and shows only confident results.But if the confidence is below 25%, it ignores it.
results = model(img_path, conf=0.25)
#هنا ياخذ اول عنصر
r = results[0]
#Boxes are the rectangles around detected parking spaces.
boxes = r.boxes
#I created two lists: one for empty spaces and one for occupied spaces.
empty_spaces = []
occupied_spaces = []
#This line gets the class names from the model.
names = model.names
#This gets the class name, either empty or occupied.Loop → get ID → convert to name
for box in boxes:
    cls_id = int(box.cls[0])
    cls_name = names[cls_id]

    x1, y1, x2, y2 = map(int, box.xyxy[0])# هذه الإحداثيات جاهزة من المودل
    conf = float(box.conf[0])

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    spot_info = {
        "box": (x1, y1, x2, y2),
        "center": (cx, cy),
        "conf": conf,
        "class": cls_name
    }
#If the class is empty, I add it to the empty list. If it is occupied, I add it to the occupied list.
    if cls_name == "empty":
        empty_spaces.append(spot_info)
    elif cls_name == "occupied":
        occupied_spaces.append(spot_info)

print("Empty:", len(empty_spaces))#عدد العناصر يحسب
print("Occupied:", len(occupied_spaces))

best_spot = None

if len(empty_spaces) > 0:
    I #defined the entrance point manually at the bottom-left of the image.
    entrance_point = (0, img.shape[0])  #يحدد مدخا الموقف  اخترت اسفل يسار الصوره
#This function calculates the distance between each empty space and the entrance.
    def distance_to_entrance(spot):
        cx, cy = spot["center"]
        ex, ey = entrance_point
        return math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
#The closest empty space to the entrance is selected as the best spot.
    best_spot = min(empty_spaces, key=distance_to_entrance)
#I made a copy of the image to draw on it.
output = img.copy()

for spot in occupied_spaces:
    x1, y1, x2, y2 = spot["box"]
    #Occupied spaces are drawn in red.
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

for spot in empty_spaces:
    x1, y1, x2, y2 = spot["box"]
    #Empty spaces are drawn in green.
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

if best_spot is not None:
    x1, y1, x2, y2 = best_spot["box"]
    #The best spot is highlighted in yellow.
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 4)
    cv2.putText(output, "Best Spot", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#This calculates the total number of detected parking spaces.
total_spaces = len(empty_spaces) + len(occupied_spaces)
#If there is at least one empty space, the status is Parking Available. Otherwise, it is Parking Full.
if len(empty_spaces) > 0:
    status_text = "Parking Available"
else:
    status_text = "Parking Full"

#I created a new image with extra space on the right side for the information panel.
h, w, _ = output.shape
panel_width = 250
new_output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)

new_output[:] = (30, 30, 30)

new_output[:, :w] = output
#I display the number of empty spaces, occupied spaces, total spaces, and parking status in the side panel.
cv2.putText(new_output, f"Empty: {len(empty_spaces)}", (w + 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.putText(new_output, f"Occupied: {len(occupied_spaces)}", (w + 20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.putText(new_output, f"Total: {total_spaces}", (w + 20, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.putText(new_output, status_text, (w + 20, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#Finally, I display the final result and save it as result.jpg.
cv2.imshow("Smart Driver Assistant", new_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
#I display the final result, then save the output image to the system.
cv2.imwrite("result.jpg", new_output)
print("Saved result as result.jpg")