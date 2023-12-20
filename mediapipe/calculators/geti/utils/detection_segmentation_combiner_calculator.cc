#include "detection_segmentation_combiner_calculator.h"

#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionSegmentationCombinerCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<GetiDetectedObject>();
  cc->Inputs().Tag("SEGMENTATION").Set<SegmentationResult>();
  cc->Outputs().Tag("DETECTION_SEGMENTATIONS").Set<DetectionSegmentation>();

  return absl::OkStatus();
}
absl::Status DetectionSegmentationCombinerCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionSegmentationCombinerCalculator::Process(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<GetiDetectedObject>();
  const auto &classifications =
      cc->Inputs().Tag("SEGMENTATION").Get<SegmentationResult>();

  auto result = std::unique_ptr<DetectionSegmentation>(
      new DetectionSegmentation{detection, classifications});

  for (auto &contour : result->segmentation_result.contours) {
    for (auto &point : contour.shape) {
      point.x += detection.shape.x;
      point.y += detection.shape.y;
    }
  }

  cc->Outputs()
      .Tag("DETECTION_SEGMENTATIONS")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionSegmentationCombinerCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionSegmentationCombinerCalculator);

}  // namespace mediapipe
