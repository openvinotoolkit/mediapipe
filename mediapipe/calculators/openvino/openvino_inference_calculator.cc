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

bool ContainsName(const std::vector<std::string> &names,
                  const std::string &name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

ov::Tensor CopyTensor(const ov::Tensor &source) {
  ov::Tensor copy(source.get_element_type(), source.get_shape());
  source.copy_to(copy);
  return copy;
}

} // namespace

// Runs inference on a pool of ov::InferRequest objects shared through the
// SESSION side packet produced by OpenVINOSessionCalculator.
//
// Inputs are either one tensor vector (TENSORS or REMOTE_TENSORS) or multiple
// individually tagged ov::Tensor streams. Named streams are bound to model
// input names with tag_to_input_tensor_names.
//
// Side inputs:
//   SESSION - std::shared_ptr<OpenVINOSession>
//
// Outputs are either one TENSORS vector or individually tagged ov::Tensor
// streams bound with tag_to_output_tensor_names.
//
//   node {
//     calculator: "OpenVINOInferenceCalculator"
//     input_side_packet: "SESSION:session"
//     input_stream: "TENSORS:input_tensors"
//     output_stream: "TENSORS:output_tensors"
//   }
class OpenVINOInferenceCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc) {
    const bool has_tensors = cc->Inputs().HasTag(kTensorsTag);
    const bool has_remote_tensors = cc->Inputs().HasTag(kRemoteTensorsTag);
    RET_CHECK(!(has_tensors && has_remote_tensors))
        << "Connect only one of TENSORS or REMOTE_TENSORS";
    const bool has_vector_input = has_tensors || has_remote_tensors;
    RET_CHECK(has_vector_input || cc->Inputs().NumEntries() > 0)
        << "OpenVINOInferenceCalculator requires input streams";
    if (has_vector_input) {
      RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
          << "A vector input cannot be combined with named input streams";
    }
    if (has_tensors) {
      cc->Inputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
    } else if (has_remote_tensors) {
      cc->Inputs().Tag(kRemoteTensorsTag).Set<std::vector<ov::RemoteTensor>>();
    } else {
      RET_CHECK(!cc->Inputs().GetTags().empty())
          << "Named tensor input streams must have tags";
      for (const std::string &tag : cc->Inputs().GetTags()) {
        RET_CHECK(!tag.empty())
            << "Named tensor input streams must use distinct tags";
        RET_CHECK_EQ(cc->Inputs().NumEntries(tag), 1)
            << "Each named tensor input tag must identify one stream: " << tag;
        cc->Inputs().Tag(tag).Set<ov::Tensor>();
      }
    }

    const bool has_vector_output = cc->Outputs().HasTag(kTensorsTag);
    RET_CHECK(has_vector_output || cc->Outputs().NumEntries() > 0)
        << "OpenVINOInferenceCalculator requires output streams";
    if (has_vector_output) {
      RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
          << "A vector output cannot be combined with named output streams";
      cc->Outputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
    } else {
      RET_CHECK(!cc->Outputs().GetTags().empty())
          << "Named tensor output streams must have tags";
      for (const std::string &tag : cc->Outputs().GetTags()) {
        RET_CHECK(!tag.empty())
            << "Named tensor output streams must use distinct tags";
        RET_CHECK_EQ(cc->Outputs().NumEntries(tag), 1)
            << "Each named tensor output tag must identify one stream: " << tag;
        cc->Outputs().Tag(tag).Set<ov::Tensor>();
      }
    }
    cc->InputSidePackets().Tag(kSessionTag).Set<OpenVINOSessionPtr>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) override {
    cc->SetOffset(TimestampDiff(0));
    session_ =
        cc->InputSidePackets().Tag(kSessionTag).Get<OpenVINOSessionPtr>();
    RET_CHECK(session_ != nullptr) << "SESSION side packet is empty";
    const auto &options = cc->Options<OpenVINOInferenceCalculatorOptions>();
    input_order_.assign(options.input_order().begin(),
                        options.input_order().end());
    output_order_.assign(options.output_order().begin(),
                         options.output_order().end());

    if (cc->Inputs().HasTag(kTensorsTag) ||
        cc->Inputs().HasTag(kRemoteTensorsTag)) {
      RET_CHECK(options.tag_to_input_tensor_names().empty())
          << "Use input_order with vector input streams";
      RET_CHECK(input_order_.empty() ||
                input_order_.size() == session_->input_names.size())
          << "input_order must contain one name per model input";
      if (!input_order_.empty()) {
        for (size_t i = 0; i < input_order_.size(); ++i) {
          RET_CHECK(ContainsName(session_->input_names, input_order_[i]))
              << "Unknown input_order tensor name: " << input_order_[i];
          RET_CHECK(std::find(input_order_.begin(), input_order_.begin() + i,
                              input_order_[i]) == input_order_.begin() + i)
              << "Duplicate input_order tensor name: " << input_order_[i];
        }
      }
    } else {
      RET_CHECK(input_order_.empty())
          << "Use tag_to_input_tensor_names with named input streams";
      const auto &bindings = options.tag_to_input_tensor_names();
      RET_CHECK_EQ(bindings.size(), cc->Inputs().GetTags().size())
          << "Provide one tag_to_input_tensor_names entry per input stream";
      std::vector<std::string> bound_names;
      for (const std::string &tag : cc->Inputs().GetTags()) {
        const auto binding = bindings.find(tag);
        RET_CHECK(binding != bindings.end())
            << "Missing input tensor mapping for tag: " << tag;
        RET_CHECK(ContainsName(session_->input_names, binding->second))
            << "Unknown model input tensor name: " << binding->second;
        RET_CHECK(!ContainsName(bound_names, binding->second))
            << "Duplicate model input tensor mapping: " << binding->second;
        bound_names.push_back(binding->second);
      }
      RET_CHECK_EQ(bound_names.size(), session_->input_names.size())
          << "Named input streams must bind every model input";
    }

    if (cc->Outputs().HasTag(kTensorsTag)) {
      RET_CHECK(options.tag_to_output_tensor_names().empty())
          << "Use output_order with vector output streams";
      for (size_t i = 0; i < output_order_.size(); ++i) {
        RET_CHECK(ContainsName(session_->output_names, output_order_[i]))
            << "Unknown output_order tensor name: " << output_order_[i];
        RET_CHECK(std::find(output_order_.begin(), output_order_.begin() + i,
                            output_order_[i]) == output_order_.begin() + i)
            << "Duplicate output_order tensor name: " << output_order_[i];
      }
    } else {
      RET_CHECK(output_order_.empty())
          << "Use tag_to_output_tensor_names with named output streams";
      const auto &bindings = options.tag_to_output_tensor_names();
      RET_CHECK_EQ(bindings.size(), cc->Outputs().GetTags().size())
          << "Provide one tag_to_output_tensor_names entry per output stream";
      std::vector<std::string> bound_names;
      for (const std::string &tag : cc->Outputs().GetTags()) {
        const auto binding = bindings.find(tag);
        RET_CHECK(binding != bindings.end())
            << "Missing output tensor mapping for tag: " << tag;
        RET_CHECK(ContainsName(session_->output_names, binding->second))
            << "Unknown model output tensor name: " << binding->second;
        RET_CHECK(!ContainsName(bound_names, binding->second))
            << "Duplicate model output tensor mapping: " << binding->second;
        bound_names.push_back(binding->second);
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext *cc) override {
    // Collect the input tensors. ov::RemoteTensor derives from ov::Tensor and
    // can be bound to an infer request through the same API.
    std::vector<ov::Tensor> inputs;
    const bool has_vector_input = cc->Inputs().HasTag(kTensorsTag) ||
                                  cc->Inputs().HasTag(kRemoteTensorsTag);
    if (cc->Inputs().HasTag(kTensorsTag)) {
      if (cc->Inputs().Tag(kTensorsTag).IsEmpty())
        return absl::OkStatus();
      inputs = cc->Inputs().Tag(kTensorsTag).Get<std::vector<ov::Tensor>>();
    } else if (cc->Inputs().HasTag(kRemoteTensorsTag)) {
      if (cc->Inputs().Tag(kRemoteTensorsTag).IsEmpty())
        return absl::OkStatus();
      const auto &remote = cc->Inputs()
                               .Tag(kRemoteTensorsTag)
                               .Get<std::vector<ov::RemoteTensor>>();
      inputs.assign(remote.begin(), remote.end());
    }

    if (!has_vector_input) {
      RET_CHECK_EQ(cc->Inputs().GetTags().size(), session_->input_names.size())
          << "Number of named input streams does not match the model";
    } else {
      RET_CHECK_EQ(inputs.size(), session_->input_names.size())
          << "Number of input tensors does not match the model";
    }

    try {
      // Borrows a free inference request from the shared queue and returns it
      // as soon as the scope ends.
      InferRequestLease lease(session_->infer_requests);
      ov::InferRequest &request = lease.infer_request();

      if (has_vector_input) {
        for (size_t i = 0; i < inputs.size(); ++i) {
          if (!input_order_.empty()) {
            request.set_tensor(input_order_[i], inputs[i]);
          } else {
            request.set_input_tensor(i, inputs[i]);
          }
        }
      } else {
        const auto &input_names =
            cc->Options<OpenVINOInferenceCalculatorOptions>()
                .tag_to_input_tensor_names();
        for (const std::string &tag : cc->Inputs().GetTags()) {
          RET_CHECK(!cc->Inputs().Tag(tag).IsEmpty())
              << "Missing input packet for tag: " << tag;
          request.set_tensor(input_names.at(tag),
                             cc->Inputs().Tag(tag).Get<ov::Tensor>());
        }
      }

      request.infer();

      if (cc->Outputs().HasTag(kTensorsTag)) {
        const std::vector<std::string> &names =
            output_order_.empty() ? session_->output_names : output_order_;
        auto outputs = std::make_unique<std::vector<ov::Tensor>>();
        outputs->reserve(names.size());
        for (size_t i = 0; i < names.size(); ++i) {
          ov::Tensor source = output_order_.empty()
                                  ? request.get_output_tensor(i)
                                  : request.get_tensor(output_order_[i]);
          outputs->push_back(CopyTensor(source));
        }
        cc->Outputs()
            .Tag(kTensorsTag)
            .Add(outputs.release(), cc->InputTimestamp());
      } else {
        const auto &output_names =
            cc->Options<OpenVINOInferenceCalculatorOptions>()
                .tag_to_output_tensor_names();
        for (const std::string &tag : cc->Outputs().GetTags()) {
          const ov::Tensor source = request.get_tensor(output_names.at(tag));
          cc->Outputs().Tag(tag).AddPacket(
              MakePacket<ov::Tensor>(CopyTensor(source))
                  .At(cc->InputTimestamp()));
        }
      }
    } catch (const std::exception &e) {
      return absl::InternalError(absl::StrCat(
          "OpenVINOInferenceCalculator inference failed: ", e.what()));
    }

    return absl::OkStatus();
  }

private:
  OpenVINOSessionPtr session_;
  std::vector<std::string> input_order_;
  std::vector<std::string> output_order_;
};

REGISTER_CALCULATOR(OpenVINOInferenceCalculator);

} // namespace mediapipe
