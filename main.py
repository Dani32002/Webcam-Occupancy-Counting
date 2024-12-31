import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math
from Conf import *
from Centroid_Info import *

def getCoordsConf(results):

    data = results[0].boxes.data.numpy().tolist()
    labels = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class']
    pd_frame = pd.DataFrame(data, columns=labels, dtype='float')
    pd_frame = pd_frame[pd_frame['class'] == 0.0]
    pd_frame.drop('class', axis=1, inplace=True)

    return pd_frame

def paint(frame):
    for id, center in centroid_info.object_centers.items():
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        (text_width, text_height), _ = cv2.getTextSize(str(id), conf.font, conf.font_size, conf.font_thickness)
        text_x = (center[0] - text_width) // 2
        text_y = (center[1] - text_height) // 2
        cv2.putText(frame, str(id), (text_x, text_y), conf.font, conf.font_size, conf.font_color, conf.font_thickness)

    cv2.line(frame, (300, 0), (300, 480), (0,0,255), 2)

    cv2.imshow("Video", frame)

def startModelCamera():
    cap = cv2.VideoCapture(conf.camera)

    model=YOLO(conf.model)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    return model, cap

def alreadyInPastFrame(new_center):

    for id, center in centroid_info.object_centers.items():
        dist = math.hypot(new_center[0] - center[0], new_center[1] - center[1])
        if dist <= conf.distance:
            return True, id
        
    return False, -1

def getCenters(pd_frame):

    activeIds = set()

    for _, row in pd_frame.iterrows():
        center_x = (int(row['x1']) + int(row['x2'])) // 2
        center_y = (int(row['x1']) + int(row['x2'])) // 2

        new_center = (center_x, center_y)

        alreadyIn = alreadyInPastFrame(new_center)

        if not alreadyIn[0]:
            centroid_info.object_centers[centroid_info.id_count] = new_center
            activeIds.add(centroid_info.id_count)
            centroid_info.id_count += 1
        else:
            centroid_info.object_centers[alreadyIn[1]] = new_center
            activeIds.add(alreadyIn[1])

    to_delete = set()
    for key in centroid_info.object_centers.keys():
        if key not in activeIds:
            to_delete.add(key)
    
    for key in to_delete:
        del centroid_info.object_centers[key]
        



def main():

    model, cap = startModelCamera()

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error retrieving frame")
            exit()

        count += 1
        if conf.skipFrames != 0 and count % conf.skipFrames != 0:
            continue

        frame = cv2.resize(frame, (600, 480))

        results=model.predict(frame, verbose=False)
        
        pd_frame = getCoordsConf(results)

        getCenters(pd_frame)

        paint(frame)
        
        #cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

conf = Conf(camera = 0, distance = 70, skipFrames = 7, model = 'yolov3-tinyu.pt')
centroid_info = Centroid_Info()

main()