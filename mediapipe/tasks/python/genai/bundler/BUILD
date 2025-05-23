# Copyright 2024 The MediaPipe Authors.
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

# Placeholder for internal Python strict library and test compatibility macro.
# Placeholder for internal Python strict test compatibility macro.

package(default_visibility = [
    "//mediapipe:__subpackages__",
    "//third_party/odml:__subpackages__",
])

py_library(
    name = "llm_bundler",
    srcs = ["llm_bundler.py"],
    deps = [
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_py_pb2",
        "//mediapipe/tasks/python/metadata/metadata_writers:model_asset_bundle_utils",
        "@com_google_sentencepiece//python:sentencepiece_processor",
    ],
)

py_test(
    name = "llm_bundler_test",
    srcs = ["llm_bundler_test.py"],
    srcs_version = "PY3",
    deps = [
        ":llm_bundler",
        "@com_google_sentencepiece//:sentencepiece_model_py_pb2",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)
