# Copyright (c) 2023 Intel Corporation
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
#

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])


mediapipe_proto_library(
    name = "openvino_converter_calculator_proto",
    srcs = ["openvino_converter_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvino_tensors_to_classification_calculator_proto",
    srcs = ["openvino_tensors_to_classification_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvino_tensors_to_detections_calculator_proto",
    srcs = ["openvino_tensors_to_detections_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "openvino_converter_calculator",
    srcs = ["openvino_converter_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_converter_calculator_cc_proto",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "@mediapipe//mediapipe/util:resource_util",
        "@linux_openvino//:openvino",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvino_tensors_to_classification_calculator",
    srcs = ["openvino_tensors_to_classification_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_tensors_to_classification_calculator_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@mediapipe//mediapipe/framework/formats:classification_cc_proto",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:location",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/util:resource_util",
        "@linux_openvino//:openvino",
        "@mediapipe//mediapipe/framework/port:file_helpers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvino_tensors_to_detections_calculator",
    srcs = ["openvino_tensors_to_detections_calculator.cc"],
    copts = [],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_tensors_to_detections_calculator_cc_proto",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@mediapipe//mediapipe/framework/deps:file_path",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:location",
        "@mediapipe//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@linux_openvino//:openvino",
    ],
    alwayslink = 1,
)
