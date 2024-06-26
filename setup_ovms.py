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
import getopt
import traceback

__version__ = '1.0'

class SetupOpenvinoModelServer():
  def __init__(self, force):
      self.build_lib = "mediapipe/models/ovms"
      self.force = force

  def run_command(self, command):
    print(command)
    if subprocess.call(command.split()) != 0:
      sys.exit(-1)

  def get_dst(self, build_lib, file):
    dst = os.path.join(build_lib + '/', file.replace("/","/1/"))
    dst_dir = os.path.dirname(file)
    # Workaround to copy every model in separate directory
    model_name = os.path.basename(file).replace(".tflite","")
    dir_name = os.path.basename(dst_dir)

    if dir_name != model_name:
        dst = dst.replace(dir_name + "/", model_name + "/")

    if 'ssdlite_object_detection' in file:
       build_file = os.path.join('mediapipe/', file)
       dst = os.path.join(build_lib + '/', file.replace("/","/1/"))
       dst = dst.replace('models/1', model_name + '/1')
    else:
       build_file = os.path.join('mediapipe/modules/', file)

    dst_dir = os.path.dirname(dst)

    return dst, dst_dir, build_file
  
  def _copy_to_build_lib_dir(self, build_lib, file):
    """Copy a file from bazel-bin to the build lib dir."""
    
    dst, dst_dir, build_file = self.get_dst(build_lib, file)
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    print("Copy to: " + dst)
    shutil.copyfile(os.path.join('bazel-bin/', build_file), dst)

  def _download_external_file(self, external_file):
    """Download an external file from GCS via Bazel."""

    build_file = os.path.join('mediapipe/modules/', external_file)
    if 'ssdlite_object_detection' in external_file:
       build_file = os.path.join('mediapipe/', external_file)
    fetch_model_command = [
        'bazel',
        'build',
        build_file,
    ]
    if subprocess.call(fetch_model_command) != 0:
      sys.exit(-1)
    self._copy_to_build_lib_dir(self.build_lib, external_file)

  def _copy_pbxt_file(self, external_file):

    file_to_copy = os.path.join('mediapipe/modules/', external_file)

    dst = os.path.join(self.build_lib + '/', external_file)
    dst_dir = os.path.dirname(external_file)

    if dst_dir == "face_detection":
       new_dst_dir = "face_detection_short_range"
       dst = dst.replace(dst_dir + "/", new_dst_dir + "/")

    if dst_dir == "pose_landmark":
       new_dst_dir = "pose_landmark_full"
       dst = dst.replace(dst_dir + "/", new_dst_dir + "/")

    if dst_dir == "hand_landmark":
       new_dst_dir = "hand_landmark_full"
       dst = dst.replace(dst_dir + "/", new_dst_dir + "/")
    
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    print("Copy to: " + dst)
    shutil.copyfile(file_to_copy, dst)

  def convert_pose(self):
    dst = "mediapipe/models/ovms/pose_detection/1/pose_detection.tflite"
    
    if os.path.exists(dst):
      if not self.force:
        print("File exists , not converting: " + dst + " use --force argument to overwrite.\n")
        return
      else:
         print("Re downloading pose model for conversion.")
         self._download_external_file('pose_detection/pose_detection.tflite')
      
    else:
       print("File not downloaded: " + dst + " Run setup_ovms.py --get_models first.")
       exit(0)
       
    print("Converting pose detection model")
    self.run_command("cp -r  " + dst +" .")
    self.run_command("tflite2tensorflow --model_path pose_detection.tflite --flatc_path flatbuffers/build/flatc --schema_path schema.fbs --output_pb")
    self.run_command("tflite2tensorflow --model_path pose_detection.tflite --flatc_path flatbuffers/build/flatc --schema_path schema.fbs --output_no_quant_float32_tflite   --output_dynamic_range_quant_tflite   --output_weight_quant_tflite   --output_float16_quant_tflite   --output_integer_quant_tflite")
    self.run_command("cp -rf saved_model/model_float32.tflite " + dst)
    self.run_command("rm -rf pose_detection.tflite")
    self.run_command("rm -rf saved_model")

  def get_graphs(self):
    external_files = [
        'face_detection/face_detection.pbtxt',
        'face_landmark/face_landmark_cpu.pbtxt',
        'hand_landmark/hand_landmark_cpu.pbtxt',
        #Not needed ?'holistic_landmark/hand_recrop_by_roi_cpu.pbtxt',
        'holistic_landmark/holistic_landmark_cpu.pbtxt',
        'pose_detection/pose_detection_cpu.pbtxt',
        'pose_landmark/pose_landmark_by_roi_cpu.pbtxt',
    ]
    for elem in external_files:
      print('coping file: %s\n' % elem)
      self._copy_pbxt_file(elem)

  def get_models(self):
    external_files = [
       # Using short range
       # 'face_detection/face_detection_full_range_sparse.tflite',
        'face_detection/face_detection_short_range.tflite',
        'face_landmark/face_landmark.tflite',
       # Model loading error
       # 'face_landmark/face_landmark_with_attention.tflite',
        'hand_landmark/hand_landmark_full.tflite',
       # Using full
       # 'hand_landmark/hand_landmark_lite.tflite',
        'holistic_landmark/hand_recrop.tflite',
        'iris_landmark/iris_landmark.tflite',
        'palm_detection/palm_detection_full.tflite',
       # Using full
       # 'palm_detection/palm_detection_lite.tflite',
       # Need to use OV version
        'pose_detection/pose_detection.tflite',
        'pose_landmark/pose_landmark_full.tflite',
       # Not working
       # 'selfie_segmentation/selfie_segmentation.tflite',
       # 'selfie_segmentation/selfie_segmentation_landscape.tflite',
        'models/ssdlite_object_detection.tflite',
    ]
    for elem in external_files:
      dst, dst_dir, build_lib = self.get_dst(self.build_lib, elem)
      if os.path.exists(dst) and not self.force:
         sys.stderr.write("file exists, not downloading: " + dst  + " use --force argument to overwrite.\n")
         continue
      print('downloading file: %s\n' % elem)
      self._download_external_file(elem)

def printUsage():
    """ Prints information about usage of commandline interface """

    print(""" Usage description:
               
               Get models required for ovms inference setup
               python setup_ovms.py --get_models
               
               Get graphs used in holistic client example from ovms repository
               python setup_ovms.py --get_graphs
          
               Convert original pose_detection tflite model - workaround for missing op in ov
               python setup_ovms.py --convert_pose
        """)

    return

def get_args(argv):
    """ Processing commandline """

    get_graphs_flag = False
    get_models_flag = False
    convert_pose = False
    force = False
    try:
        opts, vals = getopt.getopt(argv, "", ["force","convert_pose","get_graphs","get_models","help"])
    except getopt.GetoptError:
        print("ERROR: unrecognize option/missing argument/value for known option. Use --help to see list of options")
        sys.exit(2)
    for opt, val in opts:
        if opt in ("--help"):
            printUsage()
            sys.exit(0)
        elif opt in ("--get_graphs"):
          get_graphs_flag = True
        elif opt in ("--get_models"):
          get_models_flag = True
        elif opt in ("--convert_pose"):
          convert_pose = True
        elif opt in ("--force"):
          force = True

    return get_graphs_flag, get_models_flag, convert_pose, force

if __name__ == "__main__":
  get_graphs_flag, get_models_flag, convert_pose, force = get_args(sys.argv[1:])
  if get_models_flag:
    SetupOpenvinoModelServer(force).get_models()

  # Needed to call only on starting ovm holistic demo from ovms repository using ovms server standalone instance
  if get_graphs_flag:
    SetupOpenvinoModelServer(force).get_graphs()

  if convert_pose:
    SetupOpenvinoModelServer(force).convert_pose()
