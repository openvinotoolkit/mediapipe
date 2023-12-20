#include "detection_classification_result_calculator.h"

#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionClassificationResultCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<GetiDetectionResult>();
  cc->Inputs()
      .Tag("DETECTION_CLASSIFICATIONS")
      .Set<std::vector<DetectionClassification>>();
  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATION_RESULT")
      .Set<DetectionClassificationResult>();

  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Process(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<GetiDetectionResult>();

  auto result = std::make_unique<DetectionClassificationResult>();
  DetectionClassification prediction;
  prediction.detection = detection.objects.front();
  result->predictions.push_back(prediction);

  if (!cc->Inputs().Tag("DETECTION_CLASSIFICATIONS").IsEmpty()) {
    const auto &classifications =
        cc->Inputs()
            .Tag("DETECTION_CLASSIFICATIONS")
            .Get<std::vector<DetectionClassification>>();
    result.reset(new DetectionClassificationResult{classifications});
  }
  // add saliency maps for detection task here...

  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATION_RESULT")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionClassificationResultCalculator);

}  // namespace mediapipe
