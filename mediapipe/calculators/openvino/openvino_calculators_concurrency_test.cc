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

#include <cstddef>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

constexpr int kNumInputs = 10;
constexpr int kNumGraphs = 4;

// Model has two f32[1,10] inputs ("input1", "input2") and one f32[1,10]
// output equal to their element-wise sum.
constexpr char kGraphConfig[] = R"pb(
  input_stream: "input_tensors"
  node {
    calculator: "OpenVINOSessionCalculator"
    output_side_packet: "SESSION:session"
    node_options: {
      [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
        model_path: "mediapipe/calculators/openvino/testdata/add.xml"
        device: "CPU"
        num_infer_requests: 4
      }
    }
  }
  node {
    calculator: "OpenVINOInferenceCalculator"
    input_side_packet: "SESSION:session"
    input_stream: "TENSORS:input_tensors"
    output_stream: "TENSORS:output_tensors"
  }
)pb";

// Builds a f32[1, kNumInputs] tensor filled with `fill_value`.
ov::Tensor MakeInputTensor(float fill_value) {
  ov::Tensor tensor(ov::element::f32, ov::Shape{1, kNumInputs});
  float* data = tensor.data<float>();
  for (int i = 0; i < kNumInputs; ++i) {
    data[i] = fill_value + i;
  }
  return tensor;
}

// Runs one graph instance end-to-end with a single pair of inputs and
// returns the resulting sum tensor's data.
std::vector<float> RunGraphAndGetSum(float input1_fill, float input2_fill) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(kGraphConfig);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_tensors", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config));
  MP_EXPECT_OK(graph.StartRun({}));

  auto inputs = std::make_unique<std::vector<ov::Tensor>>();
  inputs->push_back(MakeInputTensor(input1_fill));
  inputs->push_back(MakeInputTensor(input2_fill));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input_tensors",
      Adopt(inputs.release()).At(Timestamp(0))));

  MP_EXPECT_OK(graph.CloseInputStream("input_tensors"));
  MP_EXPECT_OK(graph.WaitUntilDone());

  if (output_packets.size() != 1) {
    return {};
  }
  const auto& outputs =
      output_packets[0].Get<std::vector<ov::Tensor>>();
  if (outputs.size() != 1) {
    return {};
  }
  const ov::Tensor& sum = outputs[0];
  const float* sum_data = sum.data<float>();
  return std::vector<float>(sum_data, sum_data + sum.get_size());
}

// Creates kNumGraphs mediapipe graphs sharing the same compiled model (and
// its 4-request queue) through OpenVINOSessionCalculator's session registry,
// and runs them concurrently on separate threads, each with distinct inputs.
// This exercises the shared InferRequestLease queue under concurrency.
TEST(OpenVINOCalculatorsConcurrencyTest, FourGraphsFourInferRequests) {
  std::vector<std::thread> threads;
  std::vector<std::vector<float>> results(kNumGraphs);

  for (int g = 0; g < kNumGraphs; ++g) {
    threads.emplace_back([g, &results]() {
      const float input1_fill = static_cast<float>(g * 100);
      const float input2_fill = static_cast<float>(g * 10 + 1);
      results[g] = RunGraphAndGetSum(input1_fill, input2_fill);
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  for (int g = 0; g < kNumGraphs; ++g) {
    const float input1_fill = static_cast<float>(g * 100);
    const float input2_fill = static_cast<float>(g * 10 + 1);
    ASSERT_EQ(results[g].size(), static_cast<size_t>(kNumInputs))
        << "graph " << g;
    for (int i = 0; i < kNumInputs; ++i) {
      const float expected =
          (input1_fill + i) + (input2_fill + i);
      EXPECT_FLOAT_EQ(results[g][i], expected) << "graph " << g << " idx "
                                               << i;
    }
  }
}

}  // namespace
}  // namespace mediapipe
