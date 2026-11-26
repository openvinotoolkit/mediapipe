// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_CALCULATORS_GSTREAMER_GST_DETECTION_BOX_H_
#define MEDIAPIPE_CALCULATORS_GSTREAMER_GST_DETECTION_BOX_H_

#include <vector>

namespace mediapipe {

// Box coordinates are in pixels of the frame the overlay draws on.
struct GstDetectionBox {
  float score = 0.0f;
  float xmin = 0.0f;
  float ymin = 0.0f;
  float xmax = 0.0f;
  float ymax = 0.0f;
};

using GstDetectionBoxes = std::vector<GstDetectionBox>;

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_GSTREAMER_GST_DETECTION_BOX_H_
