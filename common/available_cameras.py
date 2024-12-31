import cv2

def list_available_cameras(max_cameras=10):
    available_cameras = []
    
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  
    
    print("Available Cameras:", available_cameras)
    return available_cameras



