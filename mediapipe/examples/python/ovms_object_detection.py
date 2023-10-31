import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
ovms_example = mp.solutions.ovms_example
mp_drawing = mp.solutions.drawing_utils
with ovms_example.OvmsObjectDetection() as ovms_object_detection:
        results = ovms_object_detection.process()
