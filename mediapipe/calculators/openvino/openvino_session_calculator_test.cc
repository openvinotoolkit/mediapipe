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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/sink.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

constexpr char kGraphTemplate[] = R"pb(
  input_stream: "input_tensors"
  node {
    calculator: "OpenVINOSessionCalculator"
    output_side_packet: "SESSION:session"
    node_options: {
      [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
        %s
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

std::vector<ov::Tensor> RunGraph(const std::string& session_options,
                                 std::vector<ov::Tensor> inputs) {
  const std::string graph_text = absl::StrFormat(kGraphTemplate, session_options);
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(graph_text);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_tensors", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input_tensors",
      MakePacket<std::vector<ov::Tensor>>(std::move(inputs)).At(Timestamp(0))));
  MP_EXPECT_OK(graph.CloseInputStream("input_tensors"));
  MP_EXPECT_OK(graph.WaitUntilDone());

  if (output_packets.size() != 1) return {};
  return output_packets[0].Get<std::vector<ov::Tensor>>();
}

ov::Tensor MakeFilledTensor(const ov::Shape& shape, float start) {
  ov::Tensor tensor(ov::element::f32, shape);
  float* data = tensor.data<float>();
  for (size_t i = 0; i < tensor.get_size(); ++i) data[i] = start + i;
  return tensor;
}

TEST(OpenVINOSessionCalculatorTest, AppliesNamedShapeAndIgnoresBatchSize) {
  constexpr char kOptions[] = R"pb(
    model_path: "mediapipe/calculators/openvino/testdata/add.xml"
    device: "CPU"
    shape: "{\"input1\":\"(2,10)\",\"input2\":\"(2,10)\"}"
    batch_size: "not-used"
    num_infer_requests: 1
  )pb";
  std::vector<ov::Tensor> outputs =
      RunGraph(kOptions, {MakeFilledTensor({2, 10}, 1.0f),
                          MakeFilledTensor({2, 10}, 100.0f)});

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].get_shape(), (ov::Shape{2, 10}));
  const float* data = outputs[0].data<float>();
  for (int i = 0; i < 20; ++i) EXPECT_FLOAT_EQ(data[i], 101.0f + 2 * i);
}

TEST(OpenVINOSessionCalculatorTest, AppliesInputPreprocessing) {
  constexpr char kOptions[] = R"pb(
    model_path: "mediapipe/calculators/openvino/testdata/identity.xml"
    device: "CPU"
    batch_size: "2"
    layout: "NHWC:NCHW"
    mean: "(1,2,3)"
    scale: "[2,4,8]"
    color_format: "RGB:BGR"
    precision: "uint8:fp32"
    num_infer_requests: 1
  )pb";
  ov::Tensor input(ov::element::u8, ov::Shape{2, 1, 2, 3});
  uint8_t* input_data = input.data<uint8_t>();
  for (int i = 0; i < 12; ++i) input_data[i] = static_cast<uint8_t>(i + 1);

  std::vector<ov::Tensor> outputs = RunGraph(kOptions, {std::move(input)});

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].get_shape(), (ov::Shape{2, 3, 1, 2}));
  const float* output = outputs[0].data<float>();
  const std::vector<float> expected = {
      1.0f, 2.5f, 0.0f, 0.75f, -0.25f, 0.125f,
      4.0f, 5.5f, 1.5f, 2.25f, 0.5f, 0.875f,
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i], expected[i]) << "index " << i;
  }
}

TEST(OpenVINOSessionCalculatorTest, AcceptsAutoShape) {
  constexpr char kOptions[] = R"pb(
    model_path: "mediapipe/calculators/openvino/testdata/identity.xml"
    device: "CPU"
    shape: "auto"
    num_infer_requests: 1
  )pb";
  std::vector<ov::Tensor> outputs =
      RunGraph(kOptions, {MakeFilledTensor({2, 3, 1, 2}, 1.0f)});

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].get_shape(), (ov::Shape{2, 3, 1, 2}));
}

}  // namespace
}  // namespace mediapipe
