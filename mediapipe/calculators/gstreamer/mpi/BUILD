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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# Intel Media Processing Interface: GStreamer decode/encode of NV12 frames into
# VA surfaces (Linux) or D3D11 textures (Windows) shareable with OpenVINO.
# Linux loads GStreamer at runtime with dlopen; Windows delay-loads it.
cc_library(
    name = "intel_mpi",
    srcs = ["intel_mpi.cpp"] + select({
        "@mediapipe//mediapipe:windows": [],
        "//conditions:default": ["gst_loader.cpp"],
    }),
    hdrs = [
        "imp_mpi_impl.h",
        "intel_mpi.h",
    ] + select({
        "@mediapipe//mediapipe:windows": [],
        "//conditions:default": ["gst_loader.h"],
    }),
    copts = select({
        "@mediapipe//mediapipe:windows": [
            "/EHsc",
            "/DGST_USE_UNSTABLE_API",
            "/wd4005",
            "/wd4100",
            "/wd4244",
            "/wd4267",
            "/wd4996",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@mediapipe//mediapipe:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    target_compatible_with = select({
        "//third_party:gstreamer_enabled": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//third_party:gstreamer",
        "//third_party:openvino",
    ],
)
