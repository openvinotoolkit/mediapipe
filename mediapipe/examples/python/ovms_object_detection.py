import mediapipe as mp
ovms_object_detection = mp.solutions.ovms_object_detection
with ovms_object_detection.OvmsObjectDetection(side_inputs=
        {'input_video_path':'/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4',
         'output_video_path':'/mediapipe/tested_video.mp4'}) as ovms_object_detection:
        results = ovms_object_detection.process()
