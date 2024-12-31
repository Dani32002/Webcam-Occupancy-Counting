import cv2

def list_available_cameras(max_cameras=10):
    available_cameras = []
    
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        
        # Check if the camera opened successfully
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()  # Release the camera after checking
    
    return available_cameras

# List all available cameras
cameras = list_available_cameras()

print("Available Cameras:", cameras)
