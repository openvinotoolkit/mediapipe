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

#include "detection_calculator.h"

#include <map>
#include <string>
#include <vector>

#include "test_utils.h"
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

#ifdef USE_MODELADAPTER
CalculatorGraphConfig graph_config =
    ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
        R"pb(
          input_stream: "input"
          input_side_packet: "model_path"
          input_side_packet: "device"
          output_stream: "output"
          node {
            calculator: "OpenVINOInferenceAdapterCalculator"
            input_side_packet: "MODEL_PATH:model_path"
            input_side_packet: "DEVICE:device"
            output_side_packet: "INFERENCE_ADAPTER:adapter"
          }
          node {
            calculator: "DetectionCalculator"
            input_side_packet: "INFERENCE_ADAPTER:adapter"
            input_stream: "IMAGE:input"
            output_stream: "DETECTIONS:output"
          }
        )pb"));
#else
CalculatorGraphConfig graph_config =
    ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
        R"pb(
          input_stream: "input"
          input_side_packet: "model_path"
          output_stream: "output"
          node {
            calculator: "DetectionCalculator"
            input_side_packet: "MODEL_PATH:model_path"
            input_stream: "IMAGE:input"
            output_stream: "DETECTIONS:output"
          }
        )pb"));
}
#endif
TEST(DetectionCalculatorTest, TestDetection) {
  const cv::Mat raw_image = cv::imread("/data/cattle.jpg");
  std::vector<Packet> output_packets;
  std::string model_path = "/data/geti/detection_atss.xml";

  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(model_path)
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));
  geti::RunGraph(mediapipe::MakePacket<cv::Mat>(raw_image), graph_config,
                 output_packets, inputSidePackets);
  ASSERT_EQ(1, output_packets.size());

  auto result = output_packets[0].Get<GetiDetectionResult>();
  auto detections = result.objects;

  ASSERT_EQ(detections.size(), 4);

  const auto &detection = detections[0];
  ASSERT_EQ(detection.label.label_id, "653b84c34e88964031d81a48");
  ASSERT_EQ(detection.label.label, "Sheep");

  const auto &cow_map = result.maps[0];
  ASSERT_EQ(cow_map.label.label_id, "653b84c34e88964031d81a47");
  ASSERT_EQ(cow_map.label.label, "Cow");

  const auto &sheep_map = result.maps[1];
  ASSERT_EQ(sheep_map.label.label_id, "653b84c34e88964031d81a48");
  ASSERT_EQ(sheep_map.label.label, "Sheep");
}

// TEST(DetectionCalculatorTest, TestTiling) {
//   const cv::Mat raw_image = cv::imread("/data/cattle.jpg");
//   std::vector<Packet> output_packets;
//   std::string model_path = "/data/geti/detection_atss_tiling.xml";
//   std::map<std::string, mediapipe::Packet> inputSidePackets;
//   inputSidePackets["model_path"] =
//       mediapipe::MakePacket<std::string>(model_path)
//           .At(mediapipe::Timestamp(0));
//   inputSidePackets["device"] =
//       mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));
//   geti::RunGraph(mediapipe::MakePacket<cv::Mat>(raw_image), graph_config,
//                  output_packets, inputSidePackets);
//
//   auto result = output_packets[0].Get<GetiDetectionResult>();
//   auto detections = result.objects;
//   ASSERT_EQ(detections.size(), 9);
//   ASSERT_EQ(result.image_size, raw_image.size());
// }

}  // namespace mediapipe
