#include "detection_classification_combiner_calculator.h"

#include "utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionClassificationCombinerCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<GetiDetectedObject>();
  cc->Inputs().Tag("CLASSIFICATION").Set<GetiClassificationResult>();
  cc->Outputs().Tag("DETECTION_CLASSIFICATIONS").Set<DetectionClassification>();

  return absl::OkStatus();
}
absl::Status DetectionClassificationCombinerCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionClassificationCombinerCalculator::Process(
    CalculatorContext *cc) {
  if (cc->Inputs().Tag("DETECTION").IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().Tag("CLASSIFICATION").IsEmpty()) {
    return absl::OkStatus();
  }
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<GetiDetectedObject>();
  const auto &classifications =
      cc->Inputs().Tag("CLASSIFICATION").Get<GetiClassificationResult>();

  auto result = std::unique_ptr<DetectionClassification>(
      new DetectionClassification{detection, classifications});

  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATIONS")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionClassificationCombinerCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionClassificationCombinerCalculator);

}  // namespace mediapipe
