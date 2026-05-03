#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/lite/interpreter.h"
// #include "mediapipe/util/tflite/config.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

namespace mediapipe {

// Converts YOLOv10 output tensors to MediaPipe Detections.
//
// YOLOv10 output tensor shape: [1, 300, 6]
// Each box: [x1, y1, x2, y2, score, class_id]
// Coordinates are in absolute pixels (0-640), need to be normalized to (0-1)
//
// Input:
//   TENSORS: Vector of TfLiteTensor
// Output:
//   DETECTIONS: Vector of Detection protos
//
// Usage:
//   node {
//     calculator: "YoloV10TensorsToDetectionsCalculator"
//     input_stream: "TENSORS:detection_tensors"
//     output_stream: "DETECTIONS:detections"
//   }

class YoloV10TensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();

     // ADD THIS to debug
    LOG(INFO) << "Number of tensors: " << tensors.size();
    for (int t = 0; t < tensors.size(); ++t) {
        LOG(INFO) << "Tensor " << t << " dims: " << tensors[t].dims->size;
        for (int d = 0; d < tensors[t].dims->size; ++d) {
        LOG(INFO) << "  dim[" << d << "] = " << tensors[t].dims->data[d];
        }
    }


    RET_CHECK(!tensors.empty()) << "No input tensors";

    // YOLOv10 has a single output tensor of shape [1, 300, 6]
    const TfLiteTensor& raw = tensors[0];
    const float* data = raw.data.f;

    // raw.dims->data = {1, 300, 6}
    // stride per box = 6 floats: [x1, y1, x2, y2, score, class_id]
    RET_CHECK_EQ(raw.dims->size, 3);
    RET_CHECK_EQ(raw.dims->data[2], 6) << "Expected 6 values per box";

    auto detections = std::make_unique<std::vector<Detection>>();

    for (int i = 0; i < num_boxes_; ++i) {
      const float* box = data + i * 6;
      
       LOG(INFO) << "Box " << i << ": "
              << "x1=" << box[0] << " y1=" << box[1]
              << " x2=" << box[2] << " y2=" << box[3]
              << " score=" << box[4] << " class=" << box[5];

      float x1       = box[0];
      float y1       = box[1];
      float x2       = box[2];
      float y2       = box[3];
      float score    = box[4];
      int   class_id = static_cast<int>(box[5]);

      if (score < min_thresh_) continue;

      // Clamp to [0, 1]
      x1 = std::max(0.0f, std::min(1.0f, x1));
      y1 = std::max(0.0f, std::min(1.0f, y1));
      x2 = std::max(0.0f, std::min(1.0f, x2));
      y2 = std::max(0.0f, std::min(1.0f, y2));

      Detection detection;

      // Set bounding box
      LocationData* location_data = detection.mutable_location_data();
      location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
      LocationData::RelativeBoundingBox* bbox =
          location_data->mutable_relative_bounding_box();
      bbox->set_xmin(x1);
      bbox->set_ymin(y1);
      bbox->set_width(x2 - x1);
      bbox->set_height(y2 - y1);

      // Set score and label
      detection.add_score(score);
      detection.add_label_id(class_id);

      detections->push_back(detection);
    }

    cc->Outputs()
        .Tag("DETECTIONS")
        .Add(detections.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
  int   num_boxes_    = 300;
  float min_thresh_   = 0.45f;
};

REGISTER_CALCULATOR(YoloV10TensorsToDetectionsCalculator);

}  // namespace mediapipe