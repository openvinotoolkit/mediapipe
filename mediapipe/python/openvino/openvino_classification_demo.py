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

"""Runs image classification on video frames through the OpenVINO calculators.

bazel run -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/python/openvino:openvino_classification_demo -- \
  --model_path=/home/dtrawins/models/resnet50/resnet50.xml \
  --input_video_path=/home/dtrawins/models/test_video.mp4
"""

import argparse

import cv2
import numpy as np

from mediapipe.python import openvino as mp_ov

GRAPH_TEMPLATE = """
input_stream: "input_tensors"
output_stream: "output_tensors"

node {{
  calculator: "OpenVINOSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {{
    [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {{
      model_path: "{model_path}"
      device: "{device}"
      num_infer_requests: {num_infer_requests}
    }}
  }}
}}

node {{
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "TENSORS:input_tensors"
  output_stream: "TENSORS:output_tensors"
}}
"""


def frame_to_nchw(frame, width, height):
  resized = cv2.resize(frame, (width, height))
  rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
  return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis, ...])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', required=True)
  parser.add_argument('--input_video_path', required=True)
  parser.add_argument('--device', default='CPU')
  parser.add_argument('--num_infer_requests', type=int, default=4)
  parser.add_argument('--input_width', type=int, default=224)
  parser.add_argument('--input_height', type=int, default=224)
  parser.add_argument('--max_frames', type=int, default=5)
  parser.add_argument('--top_k', type=int, default=5)
  args = parser.parse_args()

  graph = mp_ov.CalculatorGraph(
      graph_config=GRAPH_TEMPLATE.format(
          model_path=args.model_path,
          device=args.device,
          num_infer_requests=args.num_infer_requests))

  results = []
  graph.observe_output_stream(
      'output_tensors',
      lambda stream_name, packet: results.append(
          (packet.timestamp.value, mp_ov.get_ov_tensor_vector(packet))))
  graph.start_run()

  capture = cv2.VideoCapture(args.input_video_path)
  if not capture.isOpened():
    raise RuntimeError(f'Cannot open video {args.input_video_path}')

  frame_index = 0
  while frame_index < args.max_frames:
    ok, frame = capture.read()
    if not ok:
      break
    tensor = frame_to_nchw(frame, args.input_width, args.input_height)
    graph.add_packet_to_input_stream(
        'input_tensors',
        mp_ov.create_ov_tensor_vector([tensor]).at(frame_index))
    frame_index += 1
  capture.release()

  graph.close()

  for timestamp, tensors in results:
    for i, tensor in enumerate(tensors):
      flat = tensor.reshape(-1)
      top = np.argsort(flat)[::-1][:args.top_k]
      print(f'frame {timestamp} output[{i}] shape={tensor.shape} '
            f'dtype={tensor.dtype}')
      print('  top:', ' '.join(f'{int(c)}={flat[c]:.4f}' for c in top))


if __name__ == '__main__':
  main()
