import cv2

class Conf():

    def __init__(self, skip_frames, model, camera, confidence, distance = 100, font = cv2.FONT_HERSHEY_SIMPLEX, font_size = 1, font_color = (0, 255, 0), font_thickness = 5, backgrounds = (255, 0, 0), boxes = (0, 0, 255), label_offset = 50, fps = 30, show_camera = True, width = 320, height = 240):
        self.camera = camera
        self.distance = distance
        self.skip_frames = skip_frames
        self.model = model
        self.font = font
        self.font_size = font_size
        self.font_color = font_color 
        self.font_thickness = font_thickness
        self.confidence = confidence
        self.backgrounds = backgrounds
        self.boxes = boxes
        self.label_offset = label_offset,
        self.fps = fps,
        self.show_camera = show_camera
        self.width = width
        self.height = height