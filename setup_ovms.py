#
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

import os
import shutil
import subprocess
import sys

__version__ = '1.0'

class SetupOpenvinoModelServer():
  def __init__(self):
      self.build_lib = "mediapipe/models/ovms"

  def _copy_to_build_lib_dir(self, build_lib, file):
    """Copy a file from bazel-bin to the build lib dir."""
    dst = os.path.join(build_lib + '/', file.replace("/","/1/"))
    dst_dir = os.path.dirname(file)
    # Workaround to copy every model in separate directory
    model_name = os.path.basename(file).replace(".tflite","")
    dir_name = os.path.basename(dst_dir)
    if dir_name != model_name:
        dst = dst.replace(dir_name + "/", model_name + "/")

    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    build_file = os.path.join('mediapipe/modules/', file)
    print("Copy to: " + dst)
    shutil.copyfile(os.path.join('bazel-bin/', build_file), dst)

  def _download_external_file(self, external_file):
    """Download an external file from GCS via Bazel."""

    build_file = os.path.join('mediapipe/modules/', external_file)
    fetch_model_command = [
        'bazel',
        'build',
        build_file,
    ]
    if subprocess.call(fetch_model_command) != 0:
      sys.exit(-1)
    self._copy_to_build_lib_dir(self.build_lib, external_file)

  def run(self):
    external_files = [
       # Using short range
       # 'face_detection/face_detection_full_range_sparse.tflite',
        'face_detection/face_detection_short_range.tflite',
        'face_landmark/face_landmark.tflite',
        'face_landmark/face_landmark_with_attention.tflite',
        'hand_landmark/hand_landmark_full.tflite',
       # Using full
       # 'hand_landmark/hand_landmark_lite.tflite',
        'holistic_landmark/hand_recrop.tflite',
        'iris_landmark/iris_landmark.tflite',
        'palm_detection/palm_detection_full.tflite',
       # Using full
       # 'palm_detection/palm_detection_lite.tflite',
       # Need to use OV version
       # 'pose_detection/pose_detection.tflite',
        'pose_landmark/pose_landmark_full.tflite',
       # Not working
       # 'selfie_segmentation/selfie_segmentation.tflite',
       # 'selfie_segmentation/selfie_segmentation_landscape.tflite',
    ]
    for elem in external_files:
      sys.stderr.write('downloading file: %s\n' % elem)
      self._download_external_file(elem)

if __name__ == "__main__":
  SetupOpenvinoModelServer().run()