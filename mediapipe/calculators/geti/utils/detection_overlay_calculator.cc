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
#include "detection_overlay_calculator.h"

namespace mediapipe {

absl::Status DetectionOverlayCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
  cc->Inputs().Tag("DETECTIONS").Set<GetiDetectionResult>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status DetectionOverlayCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status DetectionOverlayCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::Process()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  // Get inputs
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();
  const auto result = cc->Inputs().Tag("DETECTIONS").Get<GetiDetectionResult>();
  const auto &detections = result.objects;

  // Apply results
  cv::Mat output_img = cvimage.clone();
  auto color = cv::Scalar(255, 0, 0);
  for (auto &obj : detections) {
    auto position = cv::Point2f(obj.shape.x, obj.shape.y - 5);
    std::ostringstream conf;
    conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100
         << '%';
    cv::rectangle(output_img, obj.shape, color, 2);
    cv::putText(output_img, obj.label.label + conf.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., cv::Scalar(255, 255, 255),
                3);
    cv::putText(output_img, obj.label.label + conf.str(), position,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1., color, 2);
  }

  cc->Outputs().Tag("IMAGE").AddPacket(
      MakePacket<cv::Mat>(output_img).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

absl::Status DetectionOverlayCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "DetectionOverlayCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionOverlayCalculator);

}  // namespace mediapipe
