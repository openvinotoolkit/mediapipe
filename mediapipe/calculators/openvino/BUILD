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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "openvino_session_calculator_proto",
    srcs = ["openvino_session_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvino_inference_calculator_proto",
    srcs = ["openvino_inference_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "openvino_session",
    hdrs = [
        "openvino_infer_request_queue.h",
        "openvino_session.h",
    ],
    deps = ["@openvino//:openvino"],
)

cc_library(
    name = "openvino_session_calculator",
    srcs = ["openvino_session_calculator.cc"],
    deps = [
        ":openvino_session",
        ":openvino_session_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_github_nlohmann_json//:json",
        "@openvino//:openvino",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvino_inference_calculator",
    srcs = ["openvino_inference_calculator.cc"],
    deps = [
        ":openvino_inference_calculator_cc_proto",
        ":openvino_session",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
        "@openvino//:openvino",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvino_calculators",
    deps = [
        ":openvino_inference_calculator",
        ":openvino_session_calculator",
    ],
)

cc_test(
    name = "openvino_calculators_concurrency_test",
    srcs = ["openvino_calculators_concurrency_test.cc"],
    data = [
        "testdata/add.bin",
        "testdata/add.xml",
    ],
    deps = [
        ":openvino_calculators",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/framework/tool:sink",
        "@openvino//:openvino",
    ],
)

cc_test(
    name = "openvino_session_calculator_test",
    srcs = ["openvino_session_calculator_test.cc"],
    data = [
        "testdata/add.bin",
        "testdata/add.xml",
        "testdata/identity.bin",
        "testdata/identity.xml",
    ],
    deps = [
        ":openvino_calculators",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/strings:str_format",
        "@openvino//:openvino",
    ],
)
