# Copyright 2024 Intel Corporation
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files(["openvino_classification.pbtxt"])

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/openvino_classification:openvino_classification
# RUN:
# bazel-bin/mediapipe/examples/desktop/openvino_classification/openvino_classification \
#   --calculator_graph_config_file=mediapipe/examples/desktop/openvino_classification/openvino_classification.pbtxt \
#   --input_video_path=/path/to/video.mp4
cc_binary(
    name = "openvino_classification",
    srcs = ["openvino_classification_main.cc"],
    deps = [
        "//mediapipe/calculators/openvino:openvino_calculators",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@openvino//:openvino",
    ],
)
