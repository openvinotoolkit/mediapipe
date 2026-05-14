#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#include <openvino/openvino.hpp>

namespace mediapipe {

// Converts YOLOv10 output OV tensors to MediaPipe Detections.
//
// YOLOv10 output tensor shape: [1, 300, 6]
// Each box: [x1, y1, x2, y2, score, class_id]
// Coordinates are normalized to (0-1)
//
// Input:
//   TENSORS: Vector of ov::Tensor
// Output:
//   DETECTIONS: Vector of Detection protos
//
// Usage:
//   node {
//     calculator: "OpenVINOYoloV10TensorsToDetectionsCalculator"
//     input_stream: "TENSORS:detection_tensors"
//     output_stream: "DETECTIONS:detections"
//   }

class OpenVINOYoloV10TensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(!cc->Inputs().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());

    if (cc->Inputs().HasTag("TENSORS")) {
      cc->Inputs().Tag("TENSORS").Set<std::vector<ov::Tensor>>();
    }
    if (cc->Outputs().HasTag("DETECTIONS")) {
      cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag("TENSORS").IsEmpty()) {
      return absl::OkStatus();
    }

    const auto& tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<ov::Tensor>>();
    RET_CHECK(!tensors.empty()) << "No input OV tensors";
    const ov::Tensor& raw = tensors[0];

    RET_CHECK(raw.get_element_type() == ov::element::f32)
        << "Expected float32 OV tensor, got: "
        << raw.get_element_type().get_type_name();

    // Validate shape [1, 300, 6]
    const auto& shape = raw.get_shape();
    RET_CHECK_EQ(shape.size(), 3)
        << "Expected 3D tensor [1, num_boxes, 6], got rank: " << shape.size();
    RET_CHECK_EQ(shape[0], 1)
        << "Expected batch size 1, got: " << shape[0];
    RET_CHECK_EQ(shape[1], (size_t)num_boxes_)
        << "Unexpected num_boxes: " << shape[1] << " expected: " << num_boxes_;
    RET_CHECK_EQ(shape[2], 6u)
        << "Expected 6 values per box [x1,y1,x2,y2,score,class], got: "
        << shape[2];

    const float* data = raw.data<float>();
    RET_CHECK(data != nullptr) << "OV tensor data pointer is null";

    // Copy immediately to avoid dangling pointer issues
    const int total_floats = num_boxes_ * 6;
    std::vector<float> data_copy(data, data + total_floats);
    const float* d = data_copy.data();

    auto output_detections = absl::make_unique<std::vector<Detection>>();

    for (int i = 0; i < num_boxes_; ++i) {
      const float* box = d + i * 6;
      float x1       = box[0];
      float y1       = box[1];
      float x2       = box[2];
      float y2       = box[3];
      float score    = box[4];
      int   class_id = static_cast<int>(box[5]);

      // YOLOv10 pads remaining boxes with zeros — stop early
      if (score == 0.0f) break;

      // Sanity check
      if (score < 0.0f || score > 1.0f) {
        LOG(WARNING) << "Skipping box " << i
                     << " with invalid score: " << score;
        continue;
      }

      if (score < min_thresh_) continue;

      LOG(INFO) << "Box " << i << ": "
                << "x1=" << x1 << " y1=" << y1
                << " x2=" << x2 << " y2=" << y2
                << " score=" << score << " class=" << class_id;

      x1 = std::max(0.0f, std::min(1.0f, x1));
      y1 = std::max(0.0f, std::min(1.0f, y1));
      x2 = std::max(0.0f, std::min(1.0f, x2));
      y2 = std::max(0.0f, std::min(1.0f, y2));

      if (x2 <= x1 || y2 <= y1) {
        LOG(WARNING) << "Skipping box " << i << " with invalid dimensions";
        continue;
      }

      Detection detection;
      LocationData* location_data = detection.mutable_location_data();
      location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
      LocationData::RelativeBoundingBox* bbox =
          location_data->mutable_relative_bounding_box();
      bbox->set_xmin(x1);
      bbox->set_ymin(y1);
      bbox->set_width(x2 - x1);
      bbox->set_height(y2 - y1);

      detection.add_score(score);
      detection.add_label_id(class_id);

      output_detections->emplace_back(detection);
    }

    if (cc->Outputs().HasTag("DETECTIONS")) {
      cc->Outputs()
          .Tag("DETECTIONS")
          .Add(output_detections.release(), cc->InputTimestamp());
    }

    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  int   num_boxes_  = 300;
  float min_thresh_ = 0.45f;
};

REGISTER_CALCULATOR(OpenVINOYoloV10TensorsToDetectionsCalculator);

}  // namespace mediapipe