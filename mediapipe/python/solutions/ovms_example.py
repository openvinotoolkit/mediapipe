# Copyright 2021 The MediaPipe Authors.
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
"""MediaPipe Face Detection."""

import enum
from typing import NamedTuple, Union

import numpy as np
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.calculators.ovms import modelapiovmsinferencecalculator_pb2
from mediapipe.calculators.ovms import modelapiovmssessioncalculator_pb2
from mediapipe.python.solution_base import SolutionBase

_FULL_GRAPH_FILE_PATH = 'mediapipe/modules/object_detection_ovms/object_detection_ovms.binarypb'

class OvmsObjectDetection(SolutionBase):
  """MediaPipe Face Detection.

  MediaPipe Face Detection processes an RGB image and returns a list of the
  detected face location data.

  Please refer to
  https://solutions.mediapipe.dev/face_detection#python-solution-api
  for usage examples.
  """
  """
  Oryginal params in desktop example
  --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_ovms1_graph.pbtxt
  --input_side_packets "input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/mediapipe/tested_video.mp4""""
  def __init__(self):
    """Initializes a MediaPipe Object Detection object.

    Args:
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
      model_selection: 0 or 1. 0 to select a short-range model that works
        best for faces within 2 meters from the camera, and 1 for a full-range
        model best for faces within 5 meters. See details in
        https://solutions.mediapipe.dev/face_detection#model_selection.
    """

    binary_graph_path = _FULL_GRAPH_FILE_PATH

    super().__init__(
        binary_graph_path=binary_graph_path,
        side_inputs=
        {'input_video_path':'/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4',
         'output_video_path':'/mediapipe/tested_video.mp4'})

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns a list of the detected face location data.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph throws any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with a "detections" field that contains a list of the
      detected face location data.
    """

    return super().process(input_data={'image': image})
