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

#include "serialization/serialization_calculators.h"

#include <nlohmann/json_fwd.hpp>

#include "kserve.h"
#include "nlohmann/json.hpp"
#include "serialization/result_serialization.h"
#include "mediapipe/calculators/geti/utils/data_structures.h"
#include "ocv_common.hpp"

namespace mediapipe {

template <class T>
absl::Status SerializationCalculator<T>::GetContract(CalculatorContract *cc) {
  LOG(INFO) << "SerializationCalculator::GetContract()";
  cc->Inputs().Tag("RESULT").Set<T>();
  cc->Inputs().Tag("REQUEST").Set<const KFSRequest *>();
  cc->Outputs().Tag("RESPONSE").Set<KFSResponse *>();

  return absl::OkStatus();
}

template <class T>
absl::Status SerializationCalculator<T>::Open(CalculatorContext *cc) {
  LOG(INFO) << "SerializationCalculator::Open()";
  return absl::OkStatus();
}

template <class T>
absl::Status SerializationCalculator<T>::Process(CalculatorContext *cc) {
  LOG(INFO) << "SerializationCalculator::Process()";
  const auto &result = cc->Inputs().Tag("RESULT").Get<T>();

  const KFSRequest *request =
      cc->Inputs().Tag("REQUEST").Get<const KFSRequest *>();
  LOG(INFO) << "KFSRequest for model " << request->model_name();
  bool include_xai = false;
  if (request->parameters().find("include_xai") != request->parameters().end())
    include_xai = request->parameters().at("include_xai").bool_param();

  auto response = std::make_unique<inference::ModelInferResponse>();
  auto data = geti::serialize(result, include_xai);
  auto param = inference::InferParameter();
  param.mutable_string_param()->assign(data.dump());
  response->mutable_parameters()->insert({"predictions", param});
  cc->Outputs()
      .Tag("RESPONSE")
      .AddPacket(MakePacket<KFSResponse *>(response.release())
                     .At(cc->InputTimestamp()));
  return absl::OkStatus();
}
template <class T>
absl::Status SerializationCalculator<T>::Close(CalculatorContext *cc) {
  LOG(INFO) << "SerializationCalculator::Close()";
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionSerializationCalculator);
REGISTER_CALCULATOR(DetectionClassificationSerializationCalculator);
REGISTER_CALCULATOR(DetectionSegmentationSerializationCalculator);
REGISTER_CALCULATOR(RotatedDetectionSerializationCalculator);
REGISTER_CALCULATOR(ClassificationSerializationCalculator);
REGISTER_CALCULATOR(SegmentationSerializationCalculator);
REGISTER_CALCULATOR(AnomalySerializationCalculator);

}  // namespace mediapipe
