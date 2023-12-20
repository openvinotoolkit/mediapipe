/*
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

#include "rotated_detection_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "utils.h"
#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {

absl::Status RotatedDetectionCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
#ifdef USE_MODELADAPTER
  cc->InputSidePackets()
      .Tag("INFERENCE_ADAPTER")
      .Set<std::shared_ptr<InferenceAdapter>>();
#else
  cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
#endif
  cc->Outputs().Tag("DETECTIONS").Set<RotatedDetectionResult>();
  return absl::OkStatus();
}

absl::Status RotatedDetectionCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));

#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();

  auto configuration = ia->getModelConfig();
  labels = geti::get_labels_from_configuration(configuration);

  model = MaskRCNNModel::create_model(ia);
#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = MaskRCNNModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status RotatedDetectionCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::Process()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  // Get image
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  // Run Inference model
  auto inference_result = model->infer(cvimage);
  auto result = add_rotated_rects(inference_result->segmentedObjects);
  std::vector<RotatedDetectedObject> objects;

  for (auto &obj : result) {
    objects.push_back({labels[obj.labelID], obj.confidence, obj.rotated_rect});
  }

  std::unique_ptr<RotatedDetectionResult> detections =
      std::make_unique<RotatedDetectionResult>();
  detections->objects = objects;
  detections->feature_vector = inference_result->feature_vector;

  cv::Rect roi(0, 0, cvimage.cols, cvimage.rows);
  for (size_t i = 0; i < inference_result->saliency_map.size(); i++) {
    detections->maps.push_back(
        {inference_result->saliency_map[i], roi, labels[i]});
  }

  cc->Outputs()
      .Tag("DETECTIONS")
      .Add(detections.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status RotatedDetectionCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "RotatedDetectionCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(RotatedDetectionCalculator);

}  // namespace mediapipe
