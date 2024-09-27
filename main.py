import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import time
import numpy as np


model = YOLO("./models/yolov10b.pt")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
cap = cv2.VideoCapture("./samples/sample.mp4")


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


count = 0
tracker = Tracker()


fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Define codec
out = cv2.VideoWriter('output_video.mov', fourcc, 30.0, (1020, 500))  # Output file, frame rate, resolution


# sample5
outside_frame = [(358, 419), (540, 303), (288, 410), (287, 411)]
inside_frame = [(758, 236), (756, 500), (1020, 500), (1016, 294)]
with open('polygons.txt', 'r') as f:
    inside_frame = eval(f.readline().strip())
    iter(f)
    outside_frame = eval(f.readline().strip())

inside = {}
entry_times = {}
counter = 0
time_spent = 0

detection_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model(frame, verbose=False)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id == 0:
                detections.append([x1, y1, x2, y2, score])
        tracker.update(frame, detections)

    for track in tracker.tracks:
        x3, y3, x4, y4 = track.bbox
        x3 = int(x3)
        y3 = int(y3)
        x4 = int(x4)
        y4 = int(y4)
        id = track.track_id
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cv2.circle(frame, (x3, y4), 7, (255, 0, 255), -1)
        cv2.circle(frame, (x4, y4), 7, (255, 0, 255), -1)
        cvzone.putTextRect(frame, f"Person-{id}", (x3, y3), 1, 1)

        outside_check = cv2.pointPolygonTest(
            np.array(outside_frame, np.int32), (x3, y4), False
        )
        inside_check = cv2.pointPolygonTest(
            np.array(inside_frame, np.int32), (x3, y4), False
        )
        if inside_check >= 0 and (id not in inside):
            counter += 1
            inside[id] = (x3, y4)
            entry_times[id] = time.time()
        elif outside_check >= 0 and (id in inside) and inside_check < 0:
            exit_time = time.time()
            time_spent = exit_time - entry_times[id]
            del inside[id]
            del entry_times[id]


    cvzone.putTextRect(frame, f"VISITED: {counter}", (50, 60), 2, 2)
    cvzone.putTextRect(frame, f"CURRENTLY INSIDE: {len(inside)}", (50, 160), 2, 2)
    cvzone.putTextRect(frame, f"TIME SPENT: {time_spent}", (50, 260), 2, 2)
    cv2.polylines(frame, [np.array(outside_frame, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(inside_frame, np.int32)], True, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    out.write(frame)
cap.release()
cv2.destroyAllWindows()

