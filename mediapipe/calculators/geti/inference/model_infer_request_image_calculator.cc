/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */

#include "model_infer_request_image_calculator.h"

namespace mediapipe {

absl::Status ModelInferRequestImageCalculator::GetContract(
    CalculatorContract *cc) {
  LOG(INFO) << "ModelInferRequestImageCalculator::GetContract()";
  cc->Inputs().Tag("REQUEST").Set<const KFSRequest *>();
  cc->Outputs().Tag("IMAGE").Set<cv::Mat>();

  return absl::OkStatus();
}

absl::Status ModelInferRequestImageCalculator::Open(CalculatorContext *cc) {
  LOG(INFO) << "ModelInferRequestImageCalculator::Open()";
  return absl::OkStatus();
}

absl::Status ModelInferRequestImageCalculator::Process(CalculatorContext *cc) {
  LOG(INFO) << "ModelInferRequestImageCalculator::Process()";
  const KFSRequest *request =
      cc->Inputs().Tag("REQUEST").Get<const KFSRequest *>();

  LOG(INFO) << "KFSRequest for model " << request->model_name();
  auto data = request->raw_input_contents().Get(0);
  std::vector<char> image_data(data.begin(), data.end());
  auto out = cv::imdecode(image_data, 1);
  cv::cvtColor(out, out, cv::COLOR_BGR2RGB);

  cc->Outputs().Tag("IMAGE").AddPacket(
      MakePacket<cv::Mat>(out).At(cc->InputTimestamp()));
  return absl::OkStatus();
}
absl::Status ModelInferRequestImageCalculator::Close(CalculatorContext *cc) {
  LOG(INFO) << "ModelInferRequestImageCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(ModelInferRequestImageCalculator);

}  // namespace mediapipe
