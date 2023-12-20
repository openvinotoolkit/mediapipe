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

#include "instance_segmentation_calculator.h"

#include <map>
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
#include "models/image_model.h"
#include "utils/data_structures.h"

namespace mediapipe {

TEST(InstanceSegmentationCalculatorTest, TestImageSegmentation) {
#ifdef USE_MODELADAPTER
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_image"
            input_side_packet: "model_path"
            input_side_packet: "device"
            output_stream: "segmentation"
            node {
              calculator: "OpenVINOInferenceAdapterCalculator"
              input_side_packet: "MODEL_PATH:model_path"
              input_side_packet: "DEVICE:device"
              output_side_packet: "INFERENCE_ADAPTER:adapter"
            }
            node {
              calculator: "InstanceSegmentationCalculator"
              input_side_packet: "INFERENCE_ADAPTER:adapter"
              input_stream: "IMAGE:input_image"
              output_stream: "RESULT:segmentation"
            }
          )pb"));
#else
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_image"
            input_side_packet: "model_path"
            output_stream: "segmentation"
            node {
              calculator: "InstanceSegmentationCalculator"
              input_side_packet: "MODEL_PATH:model_path"
              input_stream: "IMAGE:input_image"
              output_stream: "RESULT:segmentation"
            }
          )pb"));
#endif
  const cv::Mat raw_image = cv::imread("/data/cattle.jpg");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("segmentation", &graph_config, &output_packets);

  CalculatorGraph graph(graph_config);
  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(
          "/data/geti/instance_segmentation_maskrcnn_resnet50.xml")
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));

  MP_ASSERT_OK(graph.StartRun(inputSidePackets));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image",
      mediapipe::MakePacket<cv::Mat>(raw_image).At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, output_packets.size());

  auto &result = output_packets[0].Get<SegmentationResult>();
  ASSERT_EQ(18, result.contours.size());
  ASSERT_EQ(result.contours[0].label.label, "Sheep");
  ASSERT_EQ(result.contours[0].label.label_id, "653b85cb4e88964031d81b35");
}
}  // namespace mediapipe
