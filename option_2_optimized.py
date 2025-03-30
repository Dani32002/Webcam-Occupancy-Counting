import cv2
import pandas as pd
from ultralytics import YOLO
from common import Conf
import torch
from openvino.runtime import Core
import os
import numpy as np


torch.set_num_threads(4)
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

def get_coords_conf(results):

    data = results[0].boxes.data.numpy().tolist()
    labels = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class']
    pd_frame = pd.DataFrame(data, columns = labels, dtype = 'float')
    pd_frame = pd_frame[pd_frame['class'] == 0.0]
    pd_frame = pd_frame[pd_frame['confidence'] >= conf.confidence]
    pd_frame.drop('class', axis = 1, inplace = True)
    pd_frame.drop('confidence', axis = 1, inplace = True)

    return pd_frame

def paint(frame, pd_frame):
    text = f'People in frame: {str(pd_frame.shape[0])}'
    (text_width, text_height), _ = cv2.getTextSize(text, conf.font, conf.font_size, conf.font_thickness)
    cv2.rectangle(frame, (0, 0), (text_width, text_height + conf.label_offset), conf.boxes, -1)
    cv2.putText(frame, text, (0, conf.label_offset), conf.font, conf.font_size, conf.font_color, conf.font_thickness)
    for _, row in pd_frame.iterrows():
        start_point = (int(row['x1']), int(row['y1']))
        end_point = (int(row['x2']), int(row['y2']))
        cv2.rectangle(frame, start_point, end_point, conf.backgrounds, 3)

    cv2.imshow("Video", frame)

def start_model_camera():
    cap = cv2.VideoCapture(conf.camera)

    model = None
    if not os.path.exists(conf.model + "_openvino_model/"):
        model=YOLO(conf.model + ".pt")
        model.export(format="openvino") 
    
    model = YOLO(conf.model + "_openvino_model/")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, conf.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, conf.width)
    cap.set(cv2.CAP_PROP_FPS, 15)

    return model, cap

def main():

    model, cap = start_model_camera()

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error retrieving frame")
            exit()

        if conf.show_camera:
            cv2.imshow("Video", frame)

        count += 1
        if conf.skip_frames != 0 and count % conf.skip_frames != 0:
            continue


        results = model(frame)
        
        pd_frame = get_coords_conf(results)


        print(len(pd_frame), "personas")
        #paint(frame, pd_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model_ref = input("Insert the Ultralytics model string: ")
skip_frames = int(input("Pick a number of frames to skip: "))
fps = int(input("Pick a number of fps for the camera: "))
width = int(input("Pick a width for the camera: "))
height = int(input("Pick a height for the camera: "))
show_camera = True if input("Show camera? (1,0): ") == "1" else False

conf = Conf(width = width, height = height, show_camera = show_camera, fps = fps, camera = 0, distance = 120, skip_frames = skip_frames, model = model_ref, confidence = 0.5, font_color = (255, 255, 255), font_thickness = 3)

main()