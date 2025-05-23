# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "iris_depth_from_image_desktop",
    srcs = ["iris_depth_from_image_desktop.cc"],
    data = ["//mediapipe/modules/iris_landmark:iris_landmark.tflite"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/graphs/iris_tracking:iris_depth_cpu_deps",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_binary(
    name = "iris_tracking_cpu_video_input",
    data = ["//mediapipe/modules/iris_landmark:iris_landmark.tflite"],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/iris_tracking:iris_tracking_cpu_video_input_deps",
    ],
)

cc_binary(
    name = "iris_tracking_cpu",
    data = ["//mediapipe/modules/iris_landmark:iris_landmark.tflite"],
    deps = [
        "@ovms//src:ovms_lib",
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/iris_tracking:iris_tracking_cpu_deps",
    ],
)

# Linux only
cc_binary(
    name = "iris_tracking_gpu",
    data = ["//mediapipe/modules/iris_landmark:iris_landmark.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/iris_tracking:iris_tracking_gpu_deps",
    ],
)
