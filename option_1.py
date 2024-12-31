import cv2
import pandas as pd
from ultralytics import YOLO
import math
from common import *

def get_coords_conf(results):

    data = results[0].boxes.data.numpy().tolist()
    labels = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class']
    pd_frame = pd.DataFrame(data, columns = labels, dtype = 'float')
    pd_frame = pd_frame[pd_frame['class'] == 0.0]
    pd_frame = pd_frame[pd_frame['confidence'] >= conf.confidence]
    pd_frame.drop('class', axis = 1, inplace = True)
    pd_frame.drop('confidence', axis = 1, inplace = True)

    return pd_frame

def paint(frame):
    for id, center in centroid_info.object_centers.items():
        cv2.circle(frame, center, conf.font_thickness, conf.boxes, -1)
        (text_width, text_height), _ = cv2.getTextSize(str(id), conf.font, conf.font_size, conf.font_thickness)
        text_x = (center[0] - text_width) // 2
        text_y = (center[1] - text_height) // 2
        cv2.putText(frame, str(id), (text_x, text_y), conf.font, conf.font_size, conf.font_color, conf.font_thickness)

    cv2.line(frame, (300, 0), (300, 480), (0,0,255), 2)

    cv2.imshow("Video", frame)

def start_model_camera():
    cap = cv2.VideoCapture(conf.camera)

    model=YOLO(conf.model)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    return model, cap

def already_in_past_frame(new_center):

    for id, center in centroid_info.object_centers.items():
        dist = math.hypot(new_center[0] - center[0], new_center[1] - center[1])
        if dist <= conf.distance:
            return True, id
        
    return False, -1

def get_centers(pd_frame):

    active_ids = set()

    for _, row in pd_frame.iterrows():
        center_x = (int(row['x1']) + int(row['x2'])) // 2
        center_y = (int(row['x1']) + int(row['x2'])) // 2

        new_center = (center_x, center_y)

        alreadyIn = already_in_past_frame(new_center)

        if not alreadyIn[0]:
            centroid_info.object_centers[centroid_info.id_count] = new_center
            active_ids.add(centroid_info.id_count)
            centroid_info.id_count += 1
        else:
            centroid_info.object_centers[alreadyIn[1]] = new_center
            active_ids.add(alreadyIn[1])

    to_delete = set()
    for key in centroid_info.object_centers.keys():
        if key not in active_ids:
            to_delete.add(key)
    
    for key in to_delete:
        del centroid_info.object_centers[key]
        



def main():

    model, cap = start_model_camera()

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error retrieving frame")
            exit()

        count += 1
        if conf.skip_frames != 0 and count % conf.skip_frames != 0:
            continue

        #frame = cv2.resize(frame, (600, 480))

        results=model.predict(frame, verbose = False)
        
        pd_frame = get_coords_conf(results)

        get_centers(pd_frame)

        paint(frame)
        
        #cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

conf = Conf(camera = 0, skip_frames = 0, model = 'yolov8s.pt', confidence = 0.5)
centroid_info = Centroid_Info()

main()