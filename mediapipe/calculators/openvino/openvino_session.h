// Copyright 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_SESSION_H_
#define MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_SESSION_H_

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/calculators/openvino/openvino_infer_request_queue.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace openvino_calculators {

// A compiled OpenVINO model together with the pool of inference requests that
// serve it. Instances are shared (via std::shared_ptr side packets) between
// every graph that declares the same model_path/device/plugin_config, so that
// concurrently running graphs reuse one compiled model and one request queue.
struct OpenVINOSession {
  OpenVINOSession(ov::CompiledModel compiled, int nireq)
      : compiled_model(std::move(compiled)),
        infer_requests(compiled_model, nireq) {
    for (const auto& input : compiled_model.inputs()) {
      input_names.push_back(input.get_names().empty() ? std::string()
                                                      : input.get_any_name());
    }
    for (const auto& output : compiled_model.outputs()) {
      output_names.push_back(output.get_names().empty() ? std::string()
                                                        : output.get_any_name());
    }
  }

  ov::CompiledModel compiled_model;
  OVInferRequestsQueue infer_requests;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

using OpenVINOSessionPtr = std::shared_ptr<OpenVINOSession>;

}  // namespace openvino_calculators
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_SESSION_H_
