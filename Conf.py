import cv2

class Conf():

    def __init__(self, distance, skipFrames, model, camera):
        self.camera = camera
        self.distance = distance
        self.skipFrames = skipFrames
        self.model = model
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 1
        self.font_color = (0, 255, 0)  # Green color
        self.font_thickness = 5