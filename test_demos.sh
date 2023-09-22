#!/bin/bash
docker build . -t mediapipe_upstream:latest

docker run -it mediapipe_upstream:latest bash -c "bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/holistic_tracking:holistic_tracking_cpu ; GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/holistic_tracking/holistic_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_holistic_ovms.mp4" | tee test_demos.log


docker run -it mediapipe_upstream:latest bash -c "bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu && GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_face_detection_ovms.mp4" | tee -a test_demos.log

docker run -it mediapipe_upstream:latest bash -c "bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu && GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_iris_tracking_ovms.mp4" | tee -a test_demos.log


docker run -it mediapipe_upstream:latest bash -c " bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_cpu  && GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt --input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4 --output_video_path=/mediapipe/output_object.mp4" | tee -a test_demos.log

docker run -it mediapipe_upstream:latest bash -c "bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu && GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_pose_track_ovms.mp4" | tee -a test_demos.log

cat test_demos.log | grep FPS:
