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

#include "classification_calculator.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {

TEST(ClassificationCalculatorTest, TestImageClassification) {
#ifdef USE_MODELADAPTER
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_image"
            output_stream: "classification"
            node {
              calculator: "OpenVINOInferenceAdapterCalculator"
              output_side_packet: "INFERENCE_ADAPTER:adapter"
              node_options: {
                [type.googleapis.com/
                 mediapipe.OpenVINOInferenceAdapterCalculatorOptions] {
                  model_path: "/data/geti/classification_efficientnet_b0.xml"
                }
              }
            }
            node {
              calculator: "ClassificationCalculator"
              input_side_packet: "INFERENCE_ADAPTER:adapter"
              input_stream: "IMAGE:input_image"
              output_stream: "CLASSIFICATION:classification"
            }
          )pb"));
#else
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_image"
            input_side_packet: "model_path"
            output_stream: "classification"
            node {
              calculator: "ClassificationCalculator"
              input_side_packet: "MODEL_PATH:model_path"
              input_stream: "IMAGE:input_image"
              output_stream: "CLASSIFICATION:classification"
            }
          )pb"));
#endif
  const cv::Mat raw_image = cv::imread("/data/cattle.jpg");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("classification", &graph_config, &output_packets);

  CalculatorGraph graph(graph_config);
  std::map<std::string, mediapipe::Packet> inputSidePackets;

  MP_ASSERT_OK(graph.StartRun(inputSidePackets));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image",
      mediapipe::MakePacket<cv::Mat>(raw_image).At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, output_packets.size());

  auto &classification = output_packets[0].Get<GetiClassificationResult>();
  ASSERT_EQ(classification.predictions.size(), 2);

  ASSERT_EQ(classification.predictions[0].label.label, "Cow");
  ASSERT_EQ(classification.predictions[0].roi.height, raw_image.rows);
  ASSERT_EQ(classification.predictions[0].roi.width, raw_image.cols);

  const auto &cow_map = classification.maps[0];
  ASSERT_EQ(cow_map.label.label, "Cow");
  ASSERT_EQ(cow_map.label.label_id, "653bb9844e88964031d81e30");

  const auto &sheep_map = classification.maps[1];
  ASSERT_EQ(sheep_map.label.label, "Sheep");
  ASSERT_EQ(sheep_map.label.label_id, "653bb9844e88964031d81e31");
}
}  // namespace mediapipe
