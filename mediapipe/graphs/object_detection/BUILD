# Copyright 2019 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# TODO for now we use TF Tensor as input so we may want to replace those in the end to pure OV::Tensor converters
cc_library(
    name = "desktop_ovms_calculators",
    deps = [
	    "//mediapipe/calculators/ovms:ovms_calculator", # this includes both calculators
        "@mediapipe//mediapipe/calculators/core:concatenate_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "@mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
)
