#ifndef MODEL_INFER_REQUEST_IMAGE_CALCULATOR_H_
#define MODEL_INFER_REQUEST_IMAGE_CALCULATOR_H_

#include <memory>

#include "../inference/geti_calculator_base.h"
#include "../inference/kserve.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Create model infer request image calculator that transforms the
// ModelInferRequest to a cv::Mat
//
// Input side packet:
//  REQUEST
//
// Output side packet:
//  IMAGE
//

class ModelInferRequestImageCalculator : public GetiCalculatorBase {
 const size_t MIN_SIZE = 32;
 const std::string OUT_OF_BOUNDS_ERROR = "IMAGE_SIZE_OUT_OF_BOUNDS";

 public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status GetiProcess(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;

 private:
  cv::Mat load_image(const std::vector<char> &image_data);
};

}  // namespace mediapipe

#endif  // MODEL_INFER_REQUEST_IMAGE_CALCULATOR_H_
