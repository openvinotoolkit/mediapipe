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

#include "segmentation_calculator.h"

#include <memory>
#include <string>

#include "inference/utils.h"
#include "models/image_model.h"
#include "utils/data_structures.h"

namespace mediapipe {

absl::Status SegmentationCalculator::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "SegmentationCalculator::GetContract()";
  cc->Inputs().Tag("IMAGE").Set<cv::Mat>();
#ifdef USE_MODELADAPTER
  cc->InputSidePackets()
      .Tag("INFERENCE_ADAPTER")
      .Set<std::shared_ptr<InferenceAdapter>>();
#else
  cc->InputSidePackets().Tag("MODEL_PATH").Set<std::string>();
#endif
  cc->Outputs().Tag("RESULT").Set<SegmentationResult>();
  return absl::OkStatus();
}

absl::Status SegmentationCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "SegmentationCalculator::Open()";
  cc->SetOffset(TimestampDiff(0));
#ifdef USE_MODELADAPTER
  ia = cc->InputSidePackets()
           .Tag("INFERENCE_ADAPTER")
           .Get<std::shared_ptr<InferenceAdapter>>();
  model = SegmentationModel::create_model(ia);
  auto configuration = ia->getModelConfig();
  labels = geti::get_labels_from_configuration(configuration);

  for (const auto &label : labels) {
    labels_map[label.label] = label;
  }

#else
  auto model_path = cc->InputSidePackets().Tag("MODEL_PATH").Get<std::string>();
  model = SegmentationModel::create_model(model_path);
#endif

  return absl::OkStatus();
}

absl::Status SegmentationCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "SegmentationCalculator::Process()";
  if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
    return absl::OkStatus();
  }

  const cv::Mat &cvimage = cc->Inputs().Tag("IMAGE").Get<cv::Mat>();

  auto inference =
      model->infer(cvimage)->asRef<ImageResultWithSoftPrediction>();
  std::vector<cv::Mat_<std::uint8_t>> saliency_maps_split(
      inference.saliency_map.channels());
  cv::split(inference.saliency_map, saliency_maps_split);

  cv::Rect roi(0, 0, cvimage.cols, cvimage.rows);
  // Insert first, since background label is not supplied by model.xml
  std::vector<SaliencyMap> saliency_maps = {
      {saliency_maps_split[0], roi, {"None", "otx_empty_lbl"}}};

  for (size_t i = 1; i < saliency_maps_split.size(); i++) {
    saliency_maps.push_back({saliency_maps_split[i], roi, labels[i - 1]});
  }

  std::vector<GetiContour> contours = {};
  for (const auto &contour : model->getContours(inference)) {
    contours.push_back(
        {labels_map[contour.label], contour.probability, contour.shape});
  }

  std::unique_ptr<SegmentationResult> result(new SegmentationResult{
      contours, saliency_maps, inference.feature_vector});

  cc->Outputs().Tag("RESULT").Add(result.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status SegmentationCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "SegmentationCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(SegmentationCalculator);

}  // namespace mediapipe
