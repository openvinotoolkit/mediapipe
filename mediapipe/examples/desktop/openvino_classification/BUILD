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
load("@rules_python//python:defs.bzl", "py_test")
load("//third_party:copy_runtime_dlls.bzl", "copy_runtime_dlls")

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
# On Windows, use bazel-bin\...\openvino_classification.exe with the same flags.
cc_binary(
    name = "openvino_classification",
    srcs = ["openvino_classification_main.cc"],
    data = [":runtime_dlls"],
    deps = [
        "//mediapipe/calculators/openvino:openvino_calculators",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "//third_party:openvino",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# Empty outside Windows.
copy_runtime_dlls(
    name = "runtime_dlls",
    srcs = ["//third_party:windows_runtime_dlls"],
)

py_test(
    name = "openvino_classification_test",
    srcs = ["openvino_classification_test.py"],
    args = [
        "--binary=$(rlocationpath :openvino_classification)",
        "--model=$(rlocationpath //mediapipe/calculators/openvino:testdata/identity.xml)",
    ],
    data = [
        ":openvino_classification",
        "//mediapipe/calculators/openvino:testdata/identity.bin",
        "//mediapipe/calculators/openvino:testdata/identity.xml",
    ],
    deps = [
        "@openvino_pip//numpy",
        "@openvino_pip//opencv_contrib_python",
        "@rules_python//python/runfiles",
    ],
)
