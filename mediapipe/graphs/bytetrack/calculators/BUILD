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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bytetrack_calculators",
    deps = [
        "//mediapipe/calculators/util:detection_unique_id_calculator",
        "//mediapipe/calculators/util:detection_color_by_id_calculator",
        "//mediapipe/calculators/tflite:yolox_tensors_to_detections_calculator",
        "//mediapipe/calculators/openvino:openvino_yolox_tensors_to_detections_calculator",
        ":kalman_filter",
        ":kalman_matrices",
        ":strack",
        ":bytetrack_calculator",
        ":render_data_passthrough",
    ]
)

mediapipe_proto_library(
    name = "bytetrack_calculator_proto",
    visibility = ["//visibility:public"],
    srcs = ["bytetrack_calculator.proto"],
    deps=[
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ]
)

cc_library(
    name = "kalman_matrices",
    hdrs = ["kalman_matrices.h"],
    deps = ["@eigen_archive//:eigen3"],
)


cc_library(
    name = "matching_utils",
    hdrs = ["matching_utils.h"],
    deps = [
        "//mediapipe/util/tracking:box_tracker",
    ],
)

cc_library(
    name = "kalman_filter",
    srcs = ["kalman_filter.cc"],
    hdrs = ["kalman_filter.h"],
    deps = [
        ":kalman_matrices",
        "//mediapipe/framework/formats:detection_cc_proto",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "basetrack",
    srcs = ["basetrack.cc"],
    hdrs = ["basetrack.h"]
)

cc_library(
    name = "strack",
    srcs = ["strack.cc"],
    hdrs = ["strack.h"],
    deps = [
        ":kalman_filter",
        ":basetrack",
        "//mediapipe/framework/formats:detection_cc_proto",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "bytetrack_calculator",
    srcs = ["bytetrack_calculator.cc"],
    hdrs = ["bytetrack_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        ":bytetrack_calculator_cc_proto",
        "//mediapipe/framework/port:status",
        ":matching_utils",
        ":strack",
        ":kalman_filter",
        ":kalman_matrices",
         "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

# This calculator is for debugging to check whether render data is correctly passed through the graph.
cc_library(
    name = "render_data_passthrough",
    srcs = ["render_data_passthrough_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1
)
