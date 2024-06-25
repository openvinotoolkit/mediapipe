#ifndef ANOMALY_CALCULATOR_H
#define ANOMALY_CALCULATOR_H

#include <models/anomaly_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <memory>

#include "../inference/geti_calculator_base.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"
#include "../utils/data_structures.h"

namespace mediapipe {

// Runs anomaly inference on the provided image and OpenVINO model.
//
// Input:
//  IMAGE
//
// Output:
//  RESULT
//
// Input side packet:
//  INFERENCE_ADAPTER
//

class AnomalyCalculator : public GetiCalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status GetiProcess(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;

 private:
  std::shared_ptr<InferenceAdapter> ia;
  std::unique_ptr<AnomalyModel> model;
  std::string task;
  geti::Label normal_label;
  geti::Label anomalous_label;
};

}  // namespace mediapipe

#endif  // ANOMALY_CALCULATOR_H
