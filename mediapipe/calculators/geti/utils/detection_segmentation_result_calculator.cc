#include "detection_segmentation_result_calculator.h"

#include "utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionSegmentationResultCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<GetiDetectionResult>();
  cc->Inputs()
      .Tag("DETECTION_SEGMENTATIONS")
      .Set<std::vector<DetectionSegmentation>>();
  cc->Outputs()
      .Tag("DETECTION_SEGMENTATION_RESULT")
      .Set<DetectionSegmentationResult>();

  return absl::OkStatus();
}
absl::Status DetectionSegmentationResultCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionSegmentationResultCalculator::Process(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<GetiDetectionResult>();

  auto result = std::make_unique<DetectionSegmentationResult>();
  DetectionSegmentation segmentation;
  segmentation.detection_result = detection.objects.front();
  result->segmentations.push_back(segmentation);
  result->detection = detection;

  if (!cc->Inputs().Tag("DETECTION_SEGMENTATIONS").IsEmpty()) {
    const auto &segmentations = cc->Inputs()
                                    .Tag("DETECTION_SEGMENTATIONS")
                                    .Get<std::vector<DetectionSegmentation>>();
    result.reset(new DetectionSegmentationResult{detection, segmentations});
  }
  // add saliency maps for detection task here...

  cc->Outputs()
      .Tag("DETECTION_SEGMENTATION_RESULT")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionSegmentationResultCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionSegmentationResultCalculator);

}  // namespace mediapipe
