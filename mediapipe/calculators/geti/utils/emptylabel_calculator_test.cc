
/*
**
**
** Wat wil ik testne?
**
*/

#include "test_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {

const Label test_label = {"label_id", "test_label_name"};

CalculatorGraphConfig build_graph_config(std::string calculator_name) {
  auto first_part = R"(
        input_stream: "input"
        output_stream: "output"
        node {
          calculator:")";
  auto second_part = R"("
          input_stream: "PREDICTION:input"
          output_stream: "PREDICTION:output"
          node_options: {
            [type.googleapis.com/mediapipe.EmptyLabelOptions] {
              id: "777"
              label: "mytestlabel"
            }
          }
      }
  )";
  std::stringstream ss;
  ss << first_part << calculator_name << second_part << std::endl;
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(ss.str()));
}

TEST(EmptyLabelDetectionCalculatorTest, DetectionOutput) {
  std::vector<Packet> output_packets;

  auto graph_config = build_graph_config("EmptyLabelDetectionCalculator");

  GetiDetectionResult detection;
  detection.objects = {{test_label, cv::Rect2f(10, 10, 10, 10), 0.0f}};
  geti::RunGraph(MakePacket<GetiDetectionResult>(detection), graph_config,
                 output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<GetiDetectionResult>();
  auto& first_object = result.objects[0];
  ASSERT_EQ(first_object.label.label, test_label.label);
  ASSERT_EQ(first_object.label.label_id, test_label.label_id);
  ASSERT_EQ(first_object.confidence, detection.objects[0].confidence);
}

TEST(EmptyLabelDetectionCalculatorTest, NoDetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config("EmptyLabelDetectionCalculator");

  GetiDetectionResult detection;
  detection.image_size = cv::Size(256, 128);
  geti::RunGraph(MakePacket<GetiDetectionResult>(detection), graph_config,
                 output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<GetiDetectionResult>();
  auto& first_object = result.objects[0];
  ASSERT_EQ(first_object.label.label_id, "777");
  ASSERT_EQ(first_object.label.label, "mytestlabel");
  ASSERT_EQ(first_object.confidence, 0);
  ASSERT_EQ(first_object.shape.x, 0);
  ASSERT_EQ(first_object.shape.y, 0);
  ASSERT_EQ(first_object.shape.width, detection.image_size.width);
  ASSERT_EQ(first_object.shape.height, detection.image_size.height);
}

TEST(EmptyLabelSegmentationCalculatorTest, DetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config("EmptyLabelSegmentationCalculator");

  SegmentationResult segmentation = {{{test_label, 0.0f, {}}}, {}, {}};
  geti::RunGraph(MakePacket<SegmentationResult>(segmentation), graph_config,
                 output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<SegmentationResult>();

  ASSERT_EQ(result.contours[0].label.label,
            segmentation.contours[0].label.label);
  ASSERT_EQ(result.contours[0].probability,
            segmentation.contours[0].probability);
}

TEST(EmptyLabelSegmentationCalculatorTest, NoDetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config("EmptyLabelSegmentationCalculator");

  SegmentationResult segmentation;
  geti::RunGraph(MakePacket<SegmentationResult>(segmentation), graph_config,
                 output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<SegmentationResult>();

  ASSERT_EQ(result.contours[0].label.label, "mytestlabel");
  ASSERT_EQ(result.contours[0].probability, 0);
}

TEST(EmptyLabelClassificationCalculatorTest, DetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config("EmptyLabelClassificationCalculator");
  GetiClassificationResult classification;
  classification.predictions = {{test_label, 1}};
  geti::RunGraph(MakePacket<GetiClassificationResult>(classification),
                 graph_config, output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<GetiClassificationResult>();

  ASSERT_EQ(result.predictions[0].label.label, test_label.label);
  ASSERT_EQ(result.predictions[0].score, classification.predictions[0].score);
}

TEST(EmptyLabelClassificationCalculatorTest, NoDetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config("EmptyLabelClassificationCalculator");
  GetiClassificationResult classification;
  geti::RunGraph(MakePacket<GetiClassificationResult>(classification),
                 graph_config, output_packets);
  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<GetiClassificationResult>();

  ASSERT_EQ(result.predictions[0].label.label_id, "777");
  ASSERT_EQ(result.predictions[0].label.label, "mytestlabel");
  ASSERT_EQ(result.predictions[0].score, 0);
}

TEST(EmptyLabelRotatedDetectionCalculatorTest, DetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config =
      build_graph_config("EmptyLabelRotatedDetectionCalculator");
  RotatedDetectionResult detection = {{{test_label, 0.0f, {}}}, {}, {}};
  geti::RunGraph(MakePacket<RotatedDetectionResult>(detection), graph_config,
                 output_packets);
  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<RotatedDetectionResult>();
  ASSERT_EQ(result.objects[0].label.label, test_label.label);
  ASSERT_EQ(result.objects[0].confidence, detection.objects[0].confidence);
}

TEST(EmptyLabelRotatedDetectionCalculatorTest, NoDetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config =
      build_graph_config("EmptyLabelRotatedDetectionCalculator");
  RotatedDetectionResult detection;
  geti::RunGraph(MakePacket<RotatedDetectionResult>(detection), graph_config,
                 output_packets);
  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<RotatedDetectionResult>();
  ASSERT_EQ(result.objects[0].label.label, "mytestlabel");
  ASSERT_EQ(result.objects[0].confidence, 0);
}

}  // namespace mediapipe
