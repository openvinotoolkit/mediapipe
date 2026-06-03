#include "mediapipe/calculators/tflite/yolox_tensors_to_detections_calculator.pb.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

class YoloXTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(!cc->Inputs().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());
    if (cc->Inputs().HasTag("TENSORS"))
      cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
    if (cc->Outputs().HasTag("DETECTIONS"))
      cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<mediapipe::YoloXTensorsToDetectionsCalculatorOptions>();
    min_thresh_ = options.has_conf_thresh() ? options.conf_thresh() : 0.1f;
    obj_thresh_ = options.has_obj_thresh()  ? options.obj_thresh()  : 0.1f;
    input_size_ = options.has_input_size()  ? options.input_size()  : 416.0f;
    LOG(INFO) << "Thresholds: "<<min_thresh_<<", "<<obj_thresh_;
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    LOG(INFO) << "YOLOXT2D process called";
    if (cc->Inputs().Tag("TENSORS").IsEmpty())
      return absl::OkStatus();

    const auto& tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();
    RET_CHECK(!tensors.empty());

    const TfLiteTensor& raw = tensors[0];
    RET_CHECK_EQ(raw.type, kTfLiteFloat32);
    RET_CHECK_EQ(raw.dims->size, 3);
    RET_CHECK_EQ(raw.dims->data[0], 1);
    RET_CHECK_EQ(raw.dims->data[1], num_attrs_);
    RET_CHECK_EQ(raw.dims->data[2], num_boxes_);

    const float* data = raw.data.f;
    RET_CHECK(data != nullptr);

    std::vector<float> buffer(data, data + num_attrs_ * num_boxes_);

    auto at = [&](int attr, int box) -> float {
      return buffer[attr * num_boxes_ + box];
    };

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
          if (obj < obj_thresh_) continue;

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

  absl::Status Close(CalculatorContext* cc) override {   // ✅ added for symmetry
    return absl::OkStatus();
  }

 private:
  
  const int   num_boxes_  = 3549;
  const int   num_attrs_  = 85;
  const int   num_classes_= 80;

  float input_size_;
  float min_thresh_;
  float obj_thresh_;
};

REGISTER_CALCULATOR(YoloXTensorsToDetectionsCalculator);

}  // namespace mediapipe