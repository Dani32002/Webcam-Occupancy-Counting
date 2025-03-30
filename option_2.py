import cv2
import pandas as pd
from ultralytics import YOLO
from common import Conf

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

    model=YOLO(conf.model)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    return model, cap

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

        paint(frame, pd_frame)
        
        #cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model_ref = input("Insert the Ultralytics model string: ")
skip_frames = int(input("Pick a number of frames to skip: "))

conf = Conf(camera = 0, distance = 120, skip_frames = skip_frames, model = model_ref, confidence = 0.5, font_color = (255, 255, 255), font_thickness = 3)

main()