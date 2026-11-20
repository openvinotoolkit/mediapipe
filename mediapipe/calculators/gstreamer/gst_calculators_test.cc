// Copyright 2026 Intel Corporation
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
#include <vector>

#include "mediapipe/calculators/gstreamer/gst_detection_box.h"
#include "mediapipe/calculators/gstreamer/gst_nv12_overlay_draw.h"
#include "mediapipe/calculators/openvino/openvino_test_util.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

using ::testing::HasSubstr;

TEST(SsdTensorToDetectionBoxesCalculatorTest, ScalesAndFiltersDetections) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "tensor"
    node {
      calculator: "SsdTensorToDetectionBoxesCalculator"
      input_stream: "TENSOR:tensor"
      output_stream: "DETECTION_BOXES:boxes"
      output_stream: "DETECTIONS:count"
      node_options: {
        [type.googleapis.com/mediapipe.SsdTensorToDetectionBoxesCalculatorOptions] {
          confidence_threshold: 0.5
          output_width: 100
          output_height: 50
        }
      }
    }
  )pb");
  std::vector<Packet> boxes_packets;
  std::vector<Packet> count_packets;
  tool::AddVectorSink("boxes", &config, &boxes_packets);
  tool::AddVectorSink("count", &config, &count_packets);

  ov::Tensor tensor(ov::element::f32, {1, 1, 4, 7});
  const std::vector<float> rows = {
      0, 1, 0.9f, 0.1f, 0.2f, 0.5f, 2.0f,  // kept, ymax clamped
      0, 1, 0.3f, 0.0f, 0.0f, 1.0f, 1.0f,  // below threshold
      -1, 0, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // terminator
      0, 1, 0.9f, 0.0f, 0.0f, 1.0f, 1.0f,  // after terminator
  };
  std::copy(rows.begin(), rows.end(), tensor.data<float>());

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensor", MakePacket<ov::Tensor>(tensor).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(boxes_packets.size(), 1);
  const auto& boxes = boxes_packets[0].Get<GstDetectionBoxes>();
  ASSERT_EQ(boxes.size(), 1);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.9f);
  EXPECT_FLOAT_EQ(boxes[0].xmin, 10.0f);
  EXPECT_FLOAT_EQ(boxes[0].ymin, 10.0f);
  EXPECT_FLOAT_EQ(boxes[0].xmax, 50.0f);
  EXPECT_FLOAT_EQ(boxes[0].ymax, 49.0f);
  ASSERT_EQ(count_packets.size(), 1);
  EXPECT_EQ(count_packets[0].Get<int>(), 1);
}

TEST(Nv12OverlayDrawTest, DrawsBoxBorderOnly) {
  constexpr int kWidth = 64;
  constexpr int kHeight = 64;
  std::vector<uint8_t> y(kWidth * kHeight, 0);
  std::vector<uint8_t> uv(kWidth * kHeight / 2, 0);
  const GstDetectionBoxes boxes = {{0.05f, 20.0f, 30.0f, 40.0f, 50.0f}};

  nv12overlay::DrawDetections(y.data(), kWidth, uv.data(), kWidth, kWidth,
                              kHeight, boxes);

  const nv12overlay::Color green = nv12overlay::kGreen;
  EXPECT_EQ(y[30 * kWidth + 20], green.y);
  EXPECT_EQ(y[50 * kWidth + 40], green.y);
  EXPECT_EQ(y[40 * kWidth + 30], 0);
  EXPECT_EQ(uv[15 * kWidth + 10 * 2], green.u);
  EXPECT_EQ(uv[15 * kWidth + 10 * 2 + 1], green.v);
  // Low scores do not draw the confidence bar.
  EXPECT_EQ(y[4 * kWidth + 4], 0);
}

TEST(OpenVINOInferenceCalculatorVideoFrameTest, RequiresRemoteContext) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "frame"
    node {
      calculator: "OpenVINOSessionCalculator"
      output_side_packet: "SESSION:session"
      node_options: {
        [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
          model_path: "mediapipe/calculators/openvino/testdata/identity.xml"
          device: "CPU"
          num_infer_requests: 1
        }
      }
    }
    node {
      calculator: "OpenVINOInferenceCalculator"
      input_side_packet: "SESSION:session"
      input_stream: "VIDEO_FRAME:frame"
      output_stream: "TENSORS:output"
    }
  )pb");
  ASSERT_TRUE(ResolveTestModelPath(&config));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  absl::Status status = graph.StartRun({});
  if (status.ok()) {
    graph.CloseAllInputStreams().IgnoreError();
    status = graph.WaitUntilDone();
  }
  EXPECT_THAT(status.message(), HasSubstr("remote_context"));
}

}  // namespace
}  // namespace mediapipe
