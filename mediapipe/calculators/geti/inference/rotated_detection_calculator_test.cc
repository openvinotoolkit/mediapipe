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

#include <map>
#include <string>
#include <vector>

#include "detection_calculator.h"
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
#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {

TEST(RotatedDetectionCalculatorTest, TestRotatedDetection) {
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
              calculator: "RotatedDetectionCalculator"
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
              calculator: "RotatedDetectionCalculator"
              input_side_packet: "MODEL_PATH:model_path"
              input_stream: "IMAGE:input"
              output_stream: "DETECTIONS:output"
            }
          )pb"));
}
#endif

  const cv::Mat image = cv::imread("/data/cattle.jpg");
  std::vector<Packet> output_packets;
  std::string model_path = "/data/geti/rotated_detection_maskrcnn_resnet50.xml";

  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(model_path)
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));
  geti::RunGraph(mediapipe::MakePacket<cv::Mat>(image), graph_config,
                 output_packets, inputSidePackets);
  ASSERT_EQ(1, output_packets.size());

  const auto &result = output_packets[0].Get<RotatedDetectionResult>();
  ASSERT_EQ(result.objects.size(), 9);
  const auto &obj = result.objects[0];
  ASSERT_EQ(obj.label.label, "Cow");
  ASSERT_EQ(obj.label.label_id, "653b87ce4e88964031d81d31");

  ASSERT_EQ(result.maps[0].label.label, "otx_empty_lbl");
  ASSERT_EQ(result.maps[0].label.label_id, "None");

  ASSERT_EQ(result.maps[1].label.label, "Cow");
  ASSERT_EQ(result.maps[1].label.label_id, "653b87ce4e88964031d81d31");

  ASSERT_EQ(result.maps[2].label.label, "Sheep");
  ASSERT_EQ(result.maps[2].label.label_id, "653b87ce4e88964031d81d32");
}  // namespace mediapipe

}  // namespace mediapipe
