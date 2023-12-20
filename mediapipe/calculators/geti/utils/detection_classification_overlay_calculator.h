/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */
#ifndef DETECTION_CLASSIFICATION_OVERLAY_CALCULATOR_H
#define DETECTION_CLASSIFICATION_OVERLAY_CALCULATOR_H

#include <models/input_data.h>
#include <models/results.h>

#include "data_structures.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Outputs image overlaying the detection classification task chain results
//
// Input:
//  IMAGE - cv::Mat
//  DETECTION_CLASSIFICATION - std::vector<DetectionClassificationResult>
//
// Output:
//  IMAGE - cv::Mat, Input image with applied detection bounding boxes and
//  classification information
//

class DetectionClassificationOverlayCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status Process(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;
};

}  // namespace mediapipe

#endif  // DETECTION_CLASSIFICATION_OVERLAY_CALCULATOR_H
