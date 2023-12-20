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

#include "emptylabel_calculator.h"

#include <memory>

#include "utils/data_structures.h"

namespace mediapipe {

template <class T>
absl::Status EmptyLabelCalculator<T>::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "EmptyLabelCalculator::GetContract()";
  cc->Inputs().Tag("PREDICTION").Set<T>();
  cc->Outputs().Tag("PREDICTION").Set<T>();

  return absl::OkStatus();
}

template <class T>
absl::Status EmptyLabelCalculator<T>::Open(CalculatorContext *cc) {
  LOG(INFO) << "EmptyLabelCalculator::Open()";
  return absl::OkStatus();
}

template <class T>
absl::Status EmptyLabelCalculator<T>::Process(CalculatorContext *cc) {
  LOG(INFO) << "EmptyLabelCalculator::Process()";
  const auto &prediction = cc->Inputs().Tag("PREDICTION").Get<T>();

  const auto &options = cc->Options<EmptyLabelOptions>();
  cc->Outputs()
      .Tag("PREDICTION")
      .AddPacket(MakePacket<T>(add_global_labels(prediction, options))
                     .At(cc->InputTimestamp()));

  return absl::OkStatus();
}

template <class T>
absl::Status EmptyLabelCalculator<T>::Close(CalculatorContext *cc) {
  LOG(INFO) << "EmptyLabelCalculator::Close()";
  return absl::OkStatus();
}
template <class T>
Label EmptyLabelCalculator<T>::get_label_from_options(
    const mediapipe::EmptyLabelOptions &options) {
  std::string label_name = options.label().empty() ? "empty" : options.label();
  return {options.id(), label_name};
}

template <>
GetiDetectionResult
EmptyLabelCalculator<GetiDetectionResult>::add_global_labels(
    const GetiDetectionResult &prediction,
    const mediapipe::EmptyLabelOptions &options) {
  if (prediction.objects.size() == 0) {
    auto label = get_label_from_options(options);
    GetiDetectionResult result = prediction;
    result.objects = {{label, cv::Rect2f({0, 0}, result.image_size), 0.0f}};
    return result;
  } else {
    return prediction;
  }
}

template <>
SegmentationResult EmptyLabelCalculator<SegmentationResult>::add_global_labels(
    const SegmentationResult &prediction,
    const mediapipe::EmptyLabelOptions &options) {
  if (prediction.contours.size() == 0) {
    auto label = get_label_from_options(options);

    SegmentationResult result = prediction;
    result.contours = {{label, 0, {}}};
    return result;
  } else {
    return prediction;
  }
}

template <>
GetiClassificationResult
EmptyLabelCalculator<GetiClassificationResult>::add_global_labels(
    const GetiClassificationResult &prediction,
    const mediapipe::EmptyLabelOptions &options) {
  if (prediction.predictions.size() == 0) {
    auto label = get_label_from_options(options);

    GetiClassificationResult result = prediction;
    result.predictions.push_back({label, 0.0f});
    return result;
  } else {
    return prediction;
  }
}

template <>
RotatedDetectionResult
EmptyLabelCalculator<RotatedDetectionResult>::add_global_labels(
    const RotatedDetectionResult &prediction,
    const mediapipe::EmptyLabelOptions &options) {
  if (prediction.objects.empty()) {
    auto label = get_label_from_options(options);

    RotatedDetectionResult result = prediction;
    result.objects = {{label, 0.0f, cv::RotatedRect()}};
    return result;
  } else {
    return prediction;
  }
}

REGISTER_CALCULATOR(EmptyLabelDetectionCalculator);
REGISTER_CALCULATOR(EmptyLabelClassificationCalculator);
REGISTER_CALCULATOR(EmptyLabelRotatedDetectionCalculator);
REGISTER_CALCULATOR(EmptyLabelSegmentationCalculator);

}  // namespace mediapipe
