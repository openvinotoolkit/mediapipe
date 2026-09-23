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

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pybind_extension(
    name = "_openvino_bindings",
    srcs = ["openvino_bindings.cc"],
    module_name = "_openvino_bindings",
    deps = [
        "//mediapipe/calculators/openvino:openvino_calculators",
        "//mediapipe/framework:basic_types_registration",
        "//mediapipe/framework:packet",
        "//mediapipe/python/pybind:calculator_graph",
        "//mediapipe/python/pybind:packet",
        "//mediapipe/python/pybind:resource_util",
        "//mediapipe/python/pybind:timestamp",
        "//mediapipe/python/pybind:validated_graph_config",
        "@openvino//:openvino",
    ],
)

py_library(
    name = "openvino",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [":_openvino_bindings"],
)

py_binary(
    name = "openvino_classification_demo",
    srcs = ["openvino_classification_demo.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":openvino",
        "@mediapipe_pip_deps_numpy//:pkg",
        "@mediapipe_pip_deps_opencv_contrib_python//:pkg",
    ],
)
