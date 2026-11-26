# Copyright 2026 Intel Corporation
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
load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load("//third_party:copy_runtime_dlls.bzl", "copy_runtime_dlls")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# GStreamer-backed targets are built only with --config=gst.
GSTREAMER_ONLY = select({
    "//third_party:gstreamer_enabled": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

cc_library(
    name = "gst_video_frame_packet",
    hdrs = ["gst_video_frame_packet.h"],
)

cc_library(
    name = "gst_detection_box",
    hdrs = [
        "gst_detection_box.h",
        "gst_nv12_overlay_draw.h",
    ],
)

mediapipe_proto_library(
    name = "ssd_tensor_to_detection_boxes_calculator_proto",
    srcs = ["ssd_tensor_to_detection_boxes_calculator.proto"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "ssd_tensor_to_detection_boxes_calculator",
    srcs = ["ssd_tensor_to_detection_boxes_calculator.cc"],
    deps = [
        ":gst_detection_box",
        ":ssd_tensor_to_detection_boxes_calculator_cc_proto",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gst_video_source_calculator",
    srcs = ["gst_video_source_calculator.cc"],
    target_compatible_with = GSTREAMER_ONLY,
    deps = [
        ":gst_video_frame_packet",
        "//mediapipe/calculators/gstreamer/mpi:intel_mpi",
        "//mediapipe/calculators/openvino:openvino_session",
        "//third_party:openvino",
        "@com_google_absl//absl/strings",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gst_video_sink_calculator",
    srcs = ["gst_video_sink_calculator.cc"],
    target_compatible_with = GSTREAMER_ONLY,
    deps = [
        ":gst_video_frame_packet",
        "//mediapipe/calculators/gstreamer/mpi:intel_mpi",
        "@com_google_absl//absl/strings",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gst_detection_overlay_calculator",
    srcs = ["gst_detection_overlay_calculator.cc"],
    linkopts = select({
        "@mediapipe//mediapipe:windows": [],
        "//conditions:default": ["-lva"],
    }),
    target_compatible_with = GSTREAMER_ONLY,
    deps = [
        ":gst_detection_box",
        ":gst_video_frame_packet",
        "//mediapipe/calculators/gstreamer/mpi:intel_mpi",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gst_calculators",
    target_compatible_with = GSTREAMER_ONLY,
    deps = [
        ":gst_detection_overlay_calculator",
        ":gst_video_sink_calculator",
        ":gst_video_source_calculator",
        ":ssd_tensor_to_detection_boxes_calculator",
        "//mediapipe/calculators/openvino:openvino_calculators",
    ],
)

copy_runtime_dlls(
    name = "runtime_dlls",
    srcs = ["//third_party:windows_openvino_runtime_dlls"],
)

cc_test(
    name = "gst_calculators_test",
    srcs = ["gst_calculators_test.cc"],
    data = [
        ":runtime_dlls",
        "//mediapipe/calculators/openvino:testdata/identity.bin",
        "//mediapipe/calculators/openvino:testdata/identity.xml",
    ],
    deps = [
        ":gst_detection_box",
        ":gst_video_frame_packet",
        ":ssd_tensor_to_detection_boxes_calculator",
        "//mediapipe/calculators/openvino:openvino_calculators",
        "//mediapipe/calculators/openvino:test_util",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status_matchers",
        "@mediapipe//mediapipe/framework/tool:sink",
    ],
)

# Decodes a generated H.264 clip on the GPU. Skips without a VA-capable GPU,
# GStreamer VA plugin or ffmpeg.
cc_test(
    name = "gst_video_gpu_test",
    srcs = ["gst_video_gpu_test.cc"],
    data = [":runtime_dlls"],
    target_compatible_with = GSTREAMER_ONLY,
    deps = [
        ":gst_calculators",
        ":gst_detection_box",
        ":gst_video_frame_packet",
        "//mediapipe/calculators/gstreamer/mpi:intel_mpi",
        "//third_party:openvino",
        "@com_google_absl//absl/strings",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status_matchers",
        "@mediapipe//mediapipe/framework/tool:sink",
    ],
)
