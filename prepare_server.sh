#!/bin/bash
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

MODELS_DIR=mediapipe/models/ovms/
POSE_MODEL=mediapipe/models/ovms/pose_detection/1/pose_detection.tflite
# preprare initial folders structure
mkdir -p /mediapipe/mediapipe/modules/hand_landmark/
mkdir -p ${MODELS_DIR}
# copy a text file for hand landmark module
wget -O /mediapipe/mediapipe/modules/hand_landmark/handedness.txt https://raw.githubusercontent.com/openvinotoolkit/mediapipe/v2023.1/mediapipe/modules/hand_landmark/handedness.txt
# copy ovms config including a graph definition
# download the models
curl https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite -o ${MODELS_DIR}face_detection_short_range/1/face_detection_short_range.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite -o ${MODELS_DIR}face_landmark/1/face_landmark.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite -o ${MODELS_DIR}hand_landmark_full/1/hand_landmark_full.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/hand_recrop.tflite -o ${MODELS_DIR}hand_recrop/1/hand_recrop.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite -o ${MODELS_DIR}iris_landmark/1/iris_landmark.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite -o ${MODELS_DIR}palm_detection_full/1/palm_detection_full.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite -o ${MODELS_DIR}pose_detection/1/pose_detection.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite -o ${MODELS_DIR}pose_landmark_full/1/pose_landmark_full.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/ssdlite_object_detection.tflite -o ${MODELS_DIR}ssdlite_object_detection/1/ssdlite_object_detection.tflite --create-dirs

# convert pose_detection model to the format supported by OV. It will eliminate DENDIFY layer, which is currently not supported by OpenVINO
chmod 777 ${MODELS_DIR}pose_detection/1
cp -r  ${POSE_MODEL} .
tflite2tensorflow --model_path pose_detection.tflite --flatc_path flatbuffers/build/flatc --schema_path schema.fbs --output_pb
tflite2tensorflow --model_path pose_detection.tflite --flatc_path flatbuffers/build/flatc --schema_path schema.fbs --output_no_quant_float32_tflite   --output_dynamic_range_quant_tflite   --output_weight_quant_tflite   --output_float16_quant_tflite   --output_integer_quant_tflite
cp -rf saved_model/model_float32.tflite ${POSE_MODEL}
rm -rf pose_detection.tflite
rm -rf saved_model
chmod -R 755 ${MODELS_DIR} 






