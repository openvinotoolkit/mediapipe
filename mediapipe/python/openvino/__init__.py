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

"""Python API for the MediaPipe OpenVINO calculators.

Do not import this package together with `mediapipe`: both extensions embed
their own copy of the MediaPipe framework, and pybind11 would reject the
duplicated `Packet`/`CalculatorGraph` type registrations.
"""

from mediapipe.python.openvino import _openvino_bindings

CalculatorGraph = _openvino_bindings.calculator_graph.CalculatorGraph
GraphInputStreamAddMode = (
    _openvino_bindings.calculator_graph.GraphInputStreamAddMode)
Packet = _openvino_bindings.packet.Packet
Timestamp = _openvino_bindings.timestamp.Timestamp
ValidatedGraphConfig = (
    _openvino_bindings.validated_graph_config.ValidatedGraphConfig)
create_ov_tensor_vector = _openvino_bindings.create_ov_tensor_vector
get_ov_tensor_vector = _openvino_bindings.get_ov_tensor_vector

__all__ = [
    'CalculatorGraph',
    'GraphInputStreamAddMode',
    'Packet',
    'Timestamp',
    'ValidatedGraphConfig',
    'create_ov_tensor_vector',
    'get_ov_tensor_vector',
]
