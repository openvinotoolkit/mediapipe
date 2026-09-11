#include "mediapipe/calculators/openvino/openvino_yolox_tensors_to_detections_calculator.pb.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <openvino/openvino.hpp>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Converts YOLOX output OV tensors to MediaPipe Detections.
//
// YOLOX output tensor shape: [1, 3549, 85]
// Layout: [batch, num_boxes, num_attrs]
// decode_in_inference=True: sigmoid already applied, coords already decoded
// Attributes: [cx, cy, w, h, obj_score, class_0, ..., class_79]
// Coordinates are in PIXEL space (input image 416x416), NOT normalized
//
// Input:
//   TENSORS: Vector of ov::Tensor
// Output:
//   DETECTIONS: Vector of Detection protos

class OpenVINOYoloXTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(!cc->Inputs().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());
    if (cc->Inputs().HasTag("TENSORS"))
      cc->Inputs().Tag("TENSORS").Set<std::vector<ov::Tensor>>();
    if (cc->Outputs().HasTag("DETECTIONS"))
      cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<mediapipe::OpenVINOYoloXTensorsToDetectionsCalculatorOptions>();
    min_thresh_ = options.has_conf_thresh() ? options.conf_thresh() : 0.1f;
    input_size_ = options.has_input_size() ? options.input_size() : 416.0f;
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag("TENSORS").IsEmpty())
      return absl::OkStatus();

    const auto& tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<ov::Tensor>>();
    RET_CHECK(!tensors.empty());
    const ov::Tensor& raw = tensors[0];
    RET_CHECK(raw.get_element_type() == ov::element::f32);

    const auto& shape = raw.get_shape();
    RET_CHECK_EQ(shape.size(), 3u);
    RET_CHECK_EQ(shape[0], 1u);
    // Actual layout from TFLite: [1, 85, 3549] — attr-first
    RET_CHECK_EQ(shape[1], static_cast<size_t>(num_attrs_));   // 85
    RET_CHECK_EQ(shape[2], static_cast<size_t>(num_boxes_));   // 3549

    const float* data = raw.data<float>();
    RET_CHECK(data != nullptr);

    // Accessor for [attr, box] layout
    auto at = [&](int attr, int box) -> float {
      return data[attr * num_boxes_ + box];
    };

    // Grid strides for 416x416:
    // stride 8  → 52x52 = 2704 boxes
    // stride 16 → 26x26 =  676 boxes
    // stride 32 → 13x13 =  169 boxes
    // total = 3549
    struct GridInfo { int stride; int cols; int rows; };
    const std::vector<GridInfo> grids = {
      {8,  52, 52},
      {16, 26, 26},
      {32, 13, 13},
    };

    auto output_detections = absl::make_unique<std::vector<Detection>>();

    int box_idx = 0;
    for (const auto& g : grids) {
      for (int gy = 0; gy < g.rows; ++gy) {
        for (int gx = 0; gx < g.cols; ++gx, ++box_idx) {

          // Sigmoid already baked in by TFLite Logistic ops
          float obj = at(4, box_idx);

          int   best_cls       = 0;
          float best_cls_score = 0.0f;
          for (int c = 0; c < num_classes_; ++c) {
            float s = at(5 + c, box_idx);
            if (s > best_cls_score) { best_cls_score = s; best_cls = c; }
          }

          float score = obj * best_cls_score;
          if (score < min_thresh_) continue;
          LOG(INFO)<<"CLASS: "<<best_cls<<", CLASS_SCORE: "<<best_cls_score<<", OBJECTNESS SCORE: "<<obj<< ", FINAL SCORE: "<<score;
          // Coords are raw logits — grid decode needed
          // cx, cy are offsets from grid cell origin
          // w, h are log-scale relative to stride
          float cx = (at(0, box_idx) + gx) * g.stride;
          float cy = (at(1, box_idx) + gy) * g.stride;
          float w  = std::exp(at(2, box_idx)) * g.stride;
          float h  = std::exp(at(3, box_idx)) * g.stride;

          // Normalize to [0, 1]
          float x1 = std::max(0.0f, (cx - w * 0.5f) / input_size_);
          float y1 = std::max(0.0f, (cy - h * 0.5f) / input_size_);
          float x2 = std::min(1.0f, (cx + w * 0.5f) / input_size_);
          float y2 = std::min(1.0f, (cy + h * 0.5f) / input_size_);

          if (x2 <= x1 || y2 <= y1) continue;

          Detection det;
          auto* loc = det.mutable_location_data();
          loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
          auto* bbox = loc->mutable_relative_bounding_box();
          bbox->set_xmin(x1);
          bbox->set_ymin(y1);
          bbox->set_width(x2 - x1);
          bbox->set_height(y2 - y1);
          det.add_score(score);
          det.add_label_id(best_cls);
          output_detections->emplace_back(det);
        }
      }
    }

    cc->Outputs().Tag("DETECTIONS")
        .Add(output_detections.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
 private:
  const int   num_boxes_   = 3549;
  const int   num_attrs_   = 85;
  const int   num_classes_ = 80;
  float input_size_;
  float min_thresh_;
};

REGISTER_CALCULATOR(OpenVINOYoloXTensorsToDetectionsCalculator);

}  // namespace mediapipe