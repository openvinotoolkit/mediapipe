import mediapipe as mp
ovms_object_detection = mp.solutions.ovms_object_detection
with ovms_object_detection.OvmsObjectDetection() as ovms_object_detection:
        results = ovms_object_detection.process()
