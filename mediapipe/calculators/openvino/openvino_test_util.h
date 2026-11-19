#ifndef MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_TEST_UTIL_H_
#define MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_TEST_UTIL_H_

#include <cstdlib>
#include <memory>
#include <string>

#include "mediapipe/calculators/openvino/openvino_session_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace mediapipe {

inline bool ResolveTestModelPath(CalculatorGraphConfig* config) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (!workspace) return false;
  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
      bazel::tools::cpp::runfiles::Runfiles::CreateForTest());
  if (!runfiles) return false;
  auto* node_options = config->mutable_node(0)->mutable_node_options(0);
  OpenVINOSessionCalculatorOptions options;
  if (!node_options->UnpackTo(&options)) return false;
  const std::string path = runfiles->Rlocation(
      std::string(workspace) + "/" + options.model_path());
  if (path.empty()) return false;
  options.set_model_path(path);
  node_options->PackFrom(options);
  return true;
}

}  // namespace mediapipe

#endif