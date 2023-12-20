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
#include "segmentation_overlay_calculator.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace mediapipe {

absl::Status SegmentationOverlayCalculator::GetContract(
    CalculatorContract *cc) {
  LOG(INFO) << "SegmentationOverlayCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
  cc->Inputs().Tag("SEGMENTATION_RESULT").Set<SegmentationResult>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status SegmentationOverlayCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "SegmentationOverlayCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status SegmentationOverlayCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "SegmentationOverlayCalculator::Process()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  // Get inputs
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();
  const auto &segmentation_result =
      cc->Inputs().Tag("SEGMENTATION_RESULT").Get<SegmentationResult>();

  // Apply results
  cv::Mat output_img = cvimage.clone();
  auto color = cv::Scalar(255, 0, 0);
  for (auto &obj : segmentation_result.contours) {
    auto br = cv::boundingRect(obj.shape);
    auto position =
        cv::Point2f(br.x + br.width / 2.0f, br.y + br.height / 2.0f);
    std::ostringstream conf;
    conf << ":" << std::fixed << std::setprecision(1) << obj.probability * 100
         << '%';
    std::vector<std::vector<cv::Point>> contours = {obj.shape};
    // cv::rectangle(output_img, obj, color, 2);
    cv::drawContours(output_img, contours, 0, 255);
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

absl::Status SegmentationOverlayCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "SegmentationOverlayCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(SegmentationOverlayCalculator);

}  // namespace mediapipe
