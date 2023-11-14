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
import numpy as np
import os
from typing import NamedTuple

from mediapipe.calculators.ovms import openvinoinferencecalculator_pb2
from mediapipe.calculators.ovms import openvinomodelserversessioncalculator_pb2
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python._framework_bindings import validated_graph_config

_FULL_GRAPH_FILE_PATH = 'mediapipe/modules/ovms_modules/pose_tracking_ovms.binarypb'
_POSE_RENDER_CPU_PATH = 'mediapipe/modules/ovms_modules/pose_renderer_cpu.binarypb'
_POSE_RENDER_LANDMARKS_PATH = 'mediapipe/modules/ovms_modules/pose_landmarks_to_render_data.binarypb'

class OvmsPoseTracking(SolutionBase):
  """Ovms Pose Tracking.

  Ovms Pose Tracking processes an input image frame returns output image frame
  with detected objects.
  """
  def __init__(self):
    """Initializes a Ovms Pose Tracking object.
    """

    # Initialise auxilary sub graphs
    root_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4])
    validated_graph = validated_graph_config.ValidatedGraphConfig()
    print(os.path.join(root_path, _POSE_RENDER_CPU_PATH))
    validated_graph.initialize(
          binary_graph_path=os.path.join(root_path, _POSE_RENDER_LANDMARKS_PATH))
    validated_graph.initialize(
          binary_graph_path=os.path.join(root_path, _POSE_RENDER_CPU_PATH))
  
    super().__init__(
        binary_graph_path=_FULL_GRAPH_FILE_PATH)

  # input_video is the input_stream name from the graph
  def process(self, image: np.ndarray) -> NamedTuple:
    return super().process(input_data={'input_video': image})
