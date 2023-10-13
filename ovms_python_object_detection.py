import cv2
import mediapipe as mp
ovms_example = mp.solutions.ovms_example
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = []
with ovms_example.OvmsObjectDetection() as ovms_object_detection:
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = ovms_object_detection.process()
