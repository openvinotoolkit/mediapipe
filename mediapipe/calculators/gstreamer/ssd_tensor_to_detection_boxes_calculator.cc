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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "mediapipe/calculators/gstreamer/gst_detection_box.h"
#include "mediapipe/calculators/gstreamer/ssd_tensor_to_detection_boxes_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorTag[] = "TENSOR";
constexpr char kDetectionBoxesTag[] = "DETECTION_BOXES";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr size_t kValuesPerDetection = 7;

}  // namespace

// Converts an SSD DetectionOutput tensor ([1, 1, N, 7] f32 rows of
// image_id, label, confidence, xmin, ymin, xmax, ymax with normalized
// coordinates) into pixel GstDetectionBoxes for GstDetectionOverlayCalculator.
//
// Inputs (one of):
//   TENSORS - std::vector<ov::Tensor>, the first tensor is used.
//   TENSOR  - ov::Tensor.
// Outputs:
//   DETECTION_BOXES - GstDetectionBoxes.
//   DETECTIONS      - int, number of boxes (optional).
// Input side packets (optional):
//   WIDTH, HEIGHT - int, override output_width/output_height.
class SsdTensorToDetectionBoxesCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kTensorsTag) !=
              cc->Inputs().HasTag(kTensorTag))
        << "Connect exactly one of TENSORS or TENSOR";
    if (cc->Inputs().HasTag(kTensorsTag)) {
      cc->Inputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
    } else {
      cc->Inputs().Tag(kTensorTag).Set<ov::Tensor>();
    }
    cc->Outputs().Tag(kDetectionBoxesTag).Set<GstDetectionBoxes>();
    if (cc->Outputs().HasTag(kDetectionsTag)) {
      cc->Outputs().Tag(kDetectionsTag).Set<int>();
    }
    for (const char* tag : {kWidthTag, kHeightTag}) {
      if (cc->InputSidePackets().HasTag(tag)) {
        cc->InputSidePackets().Tag(tag).Set<int>();
      }
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    const auto& options =
        cc->Options<SsdTensorToDetectionBoxesCalculatorOptions>();
    threshold_ = options.confidence_threshold();
    width_ = cc->InputSidePackets().HasTag(kWidthTag)
                 ? cc->InputSidePackets().Tag(kWidthTag).Get<int>()
                 : options.output_width();
    height_ = cc->InputSidePackets().HasTag(kHeightTag)
                  ? cc->InputSidePackets().Tag(kHeightTag).Get<int>()
                  : options.output_height();
    RET_CHECK(width_ > 0 && height_ > 0)
        << "Set output_width/output_height or WIDTH/HEIGHT";
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const ov::Tensor* tensor = nullptr;
    if (cc->Inputs().HasTag(kTensorsTag)) {
      if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) return absl::OkStatus();
      const auto& tensors =
          cc->Inputs().Tag(kTensorsTag).Get<std::vector<ov::Tensor>>();
      RET_CHECK(!tensors.empty()) << "TENSORS is empty";
      tensor = &tensors[0];
    } else {
      if (cc->Inputs().Tag(kTensorTag).IsEmpty()) return absl::OkStatus();
      tensor = &cc->Inputs().Tag(kTensorTag).Get<ov::Tensor>();
    }
    RET_CHECK(tensor->get_element_type() == ov::element::f32)
        << "SSD output must be f32";
    RET_CHECK_EQ(tensor->get_size() % kValuesPerDetection, 0u)
        << "SSD output must contain rows of 7 values";

    const float* values = tensor->data<const float>();
    const float width = static_cast<float>(width_);
    const float height = static_cast<float>(height_);
    auto boxes = std::make_unique<GstDetectionBoxes>();
    for (size_t row = 0; row < tensor->get_size();
         row += kValuesPerDetection) {
      // A negative image_id terminates the valid detections.
      if (values[row] < 0.0f) break;
      const float score = values[row + 2];
      if (score < threshold_) continue;
      boxes->push_back(
          {score, std::clamp(values[row + 3] * width, 0.0f, width - 1.0f),
           std::clamp(values[row + 4] * height, 0.0f, height - 1.0f),
           std::clamp(values[row + 5] * width, 0.0f, width - 1.0f),
           std::clamp(values[row + 6] * height, 0.0f, height - 1.0f)});
    }
    if (cc->Outputs().HasTag(kDetectionsTag)) {
      cc->Outputs().Tag(kDetectionsTag).AddPacket(
          MakePacket<int>(static_cast<int>(boxes->size()))
              .At(cc->InputTimestamp()));
    }
    cc->Outputs()
        .Tag(kDetectionBoxesTag)
        .Add(boxes.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  float threshold_ = 0.5f;
  int width_ = 0;
  int height_ = 0;
};

REGISTER_CALCULATOR(SsdTensorToDetectionBoxesCalculator);

}  // namespace mediapipe
