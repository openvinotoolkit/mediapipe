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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("//third_party:copy_runtime_dlls.bzl", "copy_runtime_dlls")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pybind_extension(
    name = "_openvino_bindings",
    srcs = ["openvino_bindings.cc"],
    data = [":runtime_dlls"],
    deps = [
        "//mediapipe/calculators/openvino:openvino_calculators",
        "@mediapipe//mediapipe/framework:basic_types_registration",
        "@mediapipe//mediapipe/framework:packet",
        "@mediapipe//mediapipe/python/pybind:calculator_graph",
        "@mediapipe//mediapipe/python/pybind:packet",
        "@mediapipe//mediapipe/python/pybind:resource_util",
        "@mediapipe//mediapipe/python/pybind:timestamp",
        "@mediapipe//mediapipe/python/pybind:validated_graph_config",
        "//third_party:openvino",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

copy_runtime_dlls(
    name = "runtime_dlls",
    srcs = ["//third_party:windows_openvino_runtime_dlls"],
)

py_library(
    name = "openvino",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    data = [":_openvino_bindings", ":runtime_dlls"],
)

py_binary(
    name = "openvino_classification_demo",
    srcs = ["openvino_classification_demo.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":openvino",
        "@openvino_pip//numpy",
        "@openvino_pip//opencv_contrib_python",
    ],
)
