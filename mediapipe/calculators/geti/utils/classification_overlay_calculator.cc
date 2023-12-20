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

#include "classification_overlay_calculator.h"

namespace mediapipe {

absl::Status ClassificationOverlayCalculator::GetContract(
    CalculatorContract *cc) {
  LOG(INFO) << "ClassificationOverlayCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
  cc->Inputs().Tag("CLASSIFICATION").Set<GetiClassificationResult>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status ClassificationOverlayCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "ClassificationOverlayCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status ClassificationOverlayCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "ClassificationOverlayCalculator::Process()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  // Get inputs
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();
  const auto &classificationResult =
      cc->Inputs().Tag("CLASSIFICATION").Get<GetiClassificationResult>();

  // Apply results
  cv::Mat output_img = cvimage.clone();
  auto color = cv::Scalar(255, 0, 0);
  auto position = cv::Point2f(10, 20);
  std::ostringstream classifications;
  for (auto &classification : classificationResult.predictions) {
    classifications << " : " << classification.label.label;
  }
  cv::putText(output_img, classifications.str(), position,
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1., cv::Scalar(255, 255, 255), 3);
  cv::putText(output_img, classifications.str(), position,
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2);

  cc->Outputs().Tag("IMAGE").AddPacket(
      MakePacket<cv::Mat>(output_img).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

absl::Status ClassificationOverlayCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "ClassificationOverlayCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ClassificationOverlayCalculator);

}  // namespace mediapipe
