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

#include "detection_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "utils.h"
#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {

absl::Status DetectionCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "DetectionCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
#ifdef USE_MODELADAPTER
  cc->InputSidePackets()
      .Tag("INFERENCE_ADAPTER")
      .Set<std::shared_ptr<InferenceAdapter>>();
#else
  cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
#endif
  cc->Outputs().Tag("DETECTIONS").Set<GetiDetectionResult>();
  return absl::OkStatus();
}

absl::Status DetectionCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "DetectionCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();
  auto configuration = ia->getModelConfig();
  labels = geti::get_labels_from_configuration(configuration);

  auto tile_size_iter = configuration.find("tile_size");
  if (tile_size_iter == configuration.end()) {
    model = DetectionModel::create_model(ia);
  } else {
    tiler = std::unique_ptr<DetectionTiler>(
        new DetectionTiler(std::move(DetectionModel::create_model(ia)), {}));
  }
#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = DetectionModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status DetectionCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "DetectionCalculator::Process()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  LOG(INFO) << "starting detection inference";

  // Get image
  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  // Run Inference model
  std::unique_ptr<GetiDetectionResult> result =
      std::make_unique<GetiDetectionResult>();
  std::unique_ptr<DetectionResult> inference_result;
  if (tiler) {
    inference_result = std::unique_ptr<DetectionResult>(
        static_cast<DetectionResult *>(tiler->run(cvimage).release()));
  } else {
    inference_result = model->infer(cvimage);
  }

  result->objects = {};
  for (auto &obj : inference_result->objects) {
    if (labels.size() > obj.labelID)
      result->objects.push_back({labels[obj.labelID], obj, obj.confidence});
  }
  result->image_size = cvimage.size();

  cv::Rect roi(0, 0, cvimage.cols, cvimage.rows);

  if (inference_result->saliency_map) {
    size_t shape_shift =
        (inference_result->saliency_map.get_shape().size() > 3) ? 1 : 0;

    inference_result->saliency_map;
    for (size_t i = 0; i < labels.size(); i++) {
      result->maps.push_back(
          {wrap_saliency_map_tensor_to_mat(inference_result->saliency_map,
                                           shape_shift, i),
           roi, labels[i]});
    }
  }
  LOG(INFO) << "completed detection inference";

  cc->Outputs().Tag("DETECTIONS").Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status DetectionCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "DetectionCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionCalculator);

}  // namespace mediapipe
