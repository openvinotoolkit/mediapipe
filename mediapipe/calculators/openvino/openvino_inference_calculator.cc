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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/openvino/openvino_infer_request_queue.h"
#include "mediapipe/calculators/openvino/openvino_inference_calculator.pb.h"
#include "mediapipe/calculators/openvino/openvino_session.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

using ::mediapipe::openvino_calculators::InferRequestLease;
using ::mediapipe::openvino_calculators::OpenVINOSession;
using ::mediapipe::openvino_calculators::OpenVINOSessionPtr;

constexpr char kSessionTag[] = "SESSION";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kRemoteTensorsTag[] = "REMOTE_TENSORS";

}  // namespace

// Runs inference on a pool of ov::InferRequest objects shared through the
// SESSION side packet produced by OpenVINOSessionCalculator.
//
// Inputs:
//   TENSORS        - std::vector<ov::Tensor>       (host tensors)
//   REMOTE_TENSORS - std::vector<ov::RemoteTensor> (device tensors)
//   Exactly one of the two input tags must be connected.
//
// Side inputs:
//   SESSION - std::shared_ptr<OpenVINOSession>
//
// Outputs:
//   TENSORS - std::vector<ov::Tensor>
//
//   node {
//     calculator: "OpenVINOInferenceCalculator"
//     input_side_packet: "SESSION:session"
//     input_stream: "TENSORS:input_tensors"
//     output_stream: "TENSORS:output_tensors"
//   }
class OpenVINOInferenceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kTensorsTag) ^
              cc->Inputs().HasTag(kRemoteTensorsTag))
        << "OpenVINOInferenceCalculator needs exactly one of TENSORS or "
           "REMOTE_TENSORS input streams";
    if (cc->Inputs().HasTag(kTensorsTag)) {
      cc->Inputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
    } else {
      cc->Inputs().Tag(kRemoteTensorsTag).Set<std::vector<ov::RemoteTensor>>();
    }
    cc->Outputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
    cc->InputSidePackets().Tag(kSessionTag).Set<OpenVINOSessionPtr>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    session_ = cc->InputSidePackets().Tag(kSessionTag).Get<OpenVINOSessionPtr>();
    RET_CHECK(session_ != nullptr) << "SESSION side packet is empty";
    const auto& options = cc->Options<OpenVINOInferenceCalculatorOptions>();
    input_order_.assign(options.input_order().begin(),
                        options.input_order().end());
    output_order_.assign(options.output_order().begin(),
                         options.output_order().end());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Collect the input tensors. ov::RemoteTensor derives from ov::Tensor and
    // can be bound to an infer request through the same API.
    std::vector<ov::Tensor> inputs;
    if (cc->Inputs().HasTag(kTensorsTag)) {
      if (cc->Inputs().Tag(kTensorsTag).IsEmpty()) return absl::OkStatus();
      inputs = cc->Inputs().Tag(kTensorsTag).Get<std::vector<ov::Tensor>>();
    } else {
      if (cc->Inputs().Tag(kRemoteTensorsTag).IsEmpty())
        return absl::OkStatus();
      const auto& remote =
          cc->Inputs().Tag(kRemoteTensorsTag).Get<std::vector<ov::RemoteTensor>>();
      inputs.assign(remote.begin(), remote.end());
    }

    RET_CHECK_EQ(inputs.size(), session_->input_names.size())
        << "Number of input tensors does not match the model";

    auto outputs = std::make_unique<std::vector<ov::Tensor>>();
    try {
      // Borrows a free inference request from the shared queue and returns it
      // as soon as the scope ends.
      InferRequestLease lease(session_->infer_requests);
      ov::InferRequest& request = lease.infer_request();

      for (size_t i = 0; i < inputs.size(); ++i) {
        if (!input_order_.empty()) {
          request.set_tensor(input_order_[i], inputs[i]);
        } else {
          request.set_input_tensor(i, inputs[i]);
        }
      }

      request.infer();

      const std::vector<std::string>& names =
          output_order_.empty() ? session_->output_names : output_order_;
      outputs->reserve(names.size());
      for (size_t i = 0; i < names.size(); ++i) {
        ov::Tensor source = output_order_.empty()
                                ? request.get_output_tensor(i)
                                : request.get_tensor(output_order_[i]);
        // The inference request is about to be reused by another packet, so
        // the results must be detached from its internal buffers.
        ov::Tensor copy(source.get_element_type(), source.get_shape());
        source.copy_to(copy);
        outputs->push_back(std::move(copy));
      }
    } catch (const std::exception& e) {
      return absl::InternalError(
          absl::StrCat("OpenVINOInferenceCalculator inference failed: ",
                       e.what()));
    }

    cc->Outputs().Tag(kTensorsTag).Add(outputs.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  OpenVINOSessionPtr session_;
  std::vector<std::string> input_order_;
  std::vector<std::string> output_order_;
};

REGISTER_CALCULATOR(OpenVINOInferenceCalculator);

}  // namespace mediapipe
