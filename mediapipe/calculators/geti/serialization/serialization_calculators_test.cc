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

#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/base.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "anomaly_calculator.h"
#include "kserve.h"
#include "mediapipe/framework/calculators/geti/inference/test_utils.h"
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
#include "mediapipe/util/image_test_utils.h"
#include "models/results.h"
#include "nlohmann/json.hpp"
#include "third_party/cpp-base64/base64.h"
#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace mediapipe {

CalculatorGraphConfig build_graph_config(std::string calculator_name) {
  auto first_part = R"(
        input_stream: "input"
        input_stream: "request"
        output_stream: "output"
        node {
          calculator:")";
  auto second_part = R"("
          input_stream: "RESULT:input"
          input_stream: "REQUEST:request"
          output_stream: "RESPONSE:output"
      }
  )";
  std::stringstream ss;
  ss << first_part << calculator_name << second_part << std::endl;
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(ss.str()));
}

struct TestData {
  std::string test_name;
  std::string type;
  nlohmann::json input;
  nlohmann::json output;
  bool include_xai;
};

inline void from_json(const nlohmann::json& j, TestData& test) {
  test.test_name = j["test_name"].template get<std::string>();
  test.type = j["type"].template get<std::string>();
  test.output = j.find("output").value();
  test.input = j.find("input").value();
  test.include_xai = false;
}

std::vector<TestData> GetTestsWithXai(const std::string& path) {
  std::ifstream input(path);
  nlohmann::json j;
  input >> j;
  input.close();
  std::vector<TestData> tests;
  for (TestData json_test : j) {
    json_test.include_xai = true;
    tests.push_back(json_test);
  }
  return tests;
}

std::vector<TestData> GetTestsWithoutXai(const std::string& path) {
  std::ifstream input(path);
  nlohmann::json j;
  input >> j;
  input.close();
  return j;
}

class CalculatorParameterizedTest : public testing::TestWithParam<TestData> {};

mediapipe::Packet build_result_packet(const nlohmann::json& input,
                                      const std::string& type) {
  if (type == "AnomalySerializationCalculator") {
    GetiAnomalyResult test_obj;
    auto image_size = input.find("image_size").value();
    Label anomalous_label{"anomalous_label",
                          input["label"].template get<std::string>()};

    std::string task = input["task"].template get<std::string>();
    int width = image_size["width"].template get<int>();
    int height = image_size["height"].template get<int>();
    float score = input["probability"].template get<float>();
    cv::Rect roi(0, 0, width, height);
    test_obj.detections.push_back({anomalous_label, roi, score});
    test_obj.maps.push_back(
        {cv::Mat::zeros(roi.size(), CV_32FC1), roi, anomalous_label});

    auto objects = input.find("objects").value();

    for (auto obj : objects) {
      cv::Rect rect(obj["x"].template get<int>(), obj["y"].template get<int>(),
                    obj["width"].template get<int>(),
                    obj["height"].template get<int>());
      if (task == "detection") {
        test_obj.detections.push_back({anomalous_label, rect, score});
      }
      if (task == "segmentation") {
        cv::Point tr(rect.x + rect.width, rect.y);
        cv::Point bl(rect.x, rect.y + rect.height);
        std::vector<cv::Point> shape = {{rect.tl(), tr, rect.br(), bl}};
        test_obj.segmentations.push_back({anomalous_label, score, shape});
      }
    }

    return mediapipe::MakePacket<GetiAnomalyResult>(test_obj);
  }

  if (type == "DetectionSerializationCalculator") {
    GetiDetectedObject test_obj;
    test_obj.label = Label{"0", input["label"].template get<std::string>()};
    test_obj.confidence = input["probability"].template get<float>();
    auto shape = input.find("shape").value();
    test_obj.shape.x = shape["x"].template get<float>();
    test_obj.shape.y = shape["y"].template get<float>();
    test_obj.shape.width = shape["width"].template get<float>();
    test_obj.shape.height = shape["height"].template get<float>();
    auto image_size = input.find("image_size").value();
    GetiDetectionResult detection_result;
    detection_result.image_size =
        cv::Size(image_size["width"].template get<int>(),
                 image_size["height"].template get<int>());

    detection_result.objects.push_back(test_obj);
    detection_result.maps.push_back(
        {cv::Mat::ones(cv::Size(7, 7), CV_8UC1),
         cv::Rect({0, 0}, detection_result.image_size), test_obj.label});

    return mediapipe::MakePacket<GetiDetectionResult>(detection_result);
  }

  if (type == "DetectionClassificationSerializationCalculator") {
    GetiDetectedObject test_obj;
    test_obj.label = Label{"0", input["label"].template get<std::string>()};
    test_obj.confidence = input["probability"].template get<float>();
    auto shape = input.find("shape").value();
    test_obj.shape.x = shape["x"].template get<float>();
    test_obj.shape.y = shape["y"].template get<float>();
    test_obj.shape.width = shape["width"].template get<float>();
    test_obj.shape.height = shape["height"].template get<float>();
    DetectionClassification detection_classification;
    detection_classification.detection = test_obj;
    auto classification = input.find("classification").value();
    detection_classification.classifications.predictions.push_back({
        Label{"0", classification["label"].template get<std::string>()},
        classification["probability"].template get<float>(),
        cv::Rect(0, 0, 0, 0),
    });

    DetectionClassificationResult test_result = {{detection_classification}};

    return mediapipe::MakePacket<DetectionClassificationResult>(test_result);
  }

  if (type == "DetectionSegmentationSerializationCalculator") {
    GetiDetectedObject detection;
    detection.label = Label{"0", input["label"].template get<std::string>()};
    detection.confidence = input["probability"].template get<float>();
    auto shape = input.find("shape").value();
    detection.shape.x = shape["x"].template get<float>();
    detection.shape.y = shape["y"].template get<float>();
    detection.shape.width = shape["width"].template get<float>();
    detection.shape.height = shape["height"].template get<float>();
    auto segmentation = input.find("segmentation").value();

    GetiContour contour;
    contour.label = Label{"0", segmentation.find("label").value()};
    contour.probability = segmentation["probability"].template get<float>();
    auto points = segmentation.find("points").value();
    for (auto& p : points) {
      contour.shape.push_back(cv::Point(p["x"].template get<float>(),
                                        p["y"].template get<float>()));
    }

    SegmentationResult segmentation_result;
    segmentation_result.contours = {contour};

    GetiDetectionResult detection_result;

    auto image_size = input.find("image_size").value();
    detection_result.image_size =
        cv::Size(image_size["width"].template get<int>(),
                 image_size["height"].template get<int>());

    detection_result.objects.push_back(detection);
    detection_result.maps.push_back(
        {cv::Mat::ones(cv::Size(7, 7), CV_8UC1),
         cv::Rect({0, 0}, detection_result.image_size), detection.label});

    DetectionSegmentation detection_segmentation;
    detection_segmentation.detection_result = detection;
    detection_segmentation.segmentation_result = segmentation_result;

    DetectionSegmentationResult test_result = {detection_result,
                                               {detection_segmentation}};
    return mediapipe::MakePacket<DetectionSegmentationResult>(test_result);
  }

  if (type == "ClassificationSerializationCalculator") {
    auto image_size = input.find("image_size").value();
    GetiClassification classification = {
        Label{"0", input.find("label").value()},
        input["score"].template get<float>(),
        cv::Rect(0, 0, image_size["width"].template get<int>(),
                 image_size["height"].template get<int>())};

    GetiClassificationResult classification_result;
    classification_result.predictions = {classification};
    classification_result.maps.push_back(
        {cv::Mat::ones(cv::Size(7, 7), CV_8UC1), classification.roi,
         classification.label});

    return mediapipe::MakePacket<GetiClassificationResult>(
        classification_result);
  }

  if (type == "SegmentationSerializationCalculator") {
    std::vector<GetiContour> contours = {};
    GetiContour test_obj;
    test_obj.label = Label{"0", input.find("label").value()};
    test_obj.probability = input["probability"].template get<float>();
    auto points = input.find("points").value();
    for (auto& p : points) {
      test_obj.shape.push_back(cv::Point(p["x"].template get<float>(),
                                         p["y"].template get<float>()));
    }
    contours.push_back(test_obj);

    auto image_size = input.find("image_size").value();
    cv::Rect roi(0, 0, image_size["width"].template get<int>(),
                 image_size["height"].template get<int>());

    SegmentationResult segmentation_result;
    segmentation_result.contours = contours;
    segmentation_result.maps.push_back(
        {cv::Mat::ones(cv::Size(7, 7), CV_8UC1), roi, test_obj.label});

    return mediapipe::MakePacket<SegmentationResult>(segmentation_result);
  }

  if (type == "RotatedDetectionSerializationCalculator") {
    std::vector<RotatedDetectedObject> input_detections = {};
    RotatedDetectedObject test_obj;
    test_obj.label = Label{"0", input["label"].template get<std::string>()};
    test_obj.confidence = input["probability"].template get<float>();
    auto shape = input.find("shape").value();

    auto center = cv::Point2f(shape["x"].template get<float>(),
                              shape["y"].template get<float>());
    auto size = cv::Point2f(shape["width"].template get<float>(),
                            shape["height"].template get<float>());
    test_obj.rotatedRectangle =
        cv::RotatedRect(center, size,

                        shape["angle"].template get<float>());
    input_detections.push_back(test_obj);

    auto image_size = input.find("image_size").value();
    cv::Rect roi(0, 0, image_size["width"].template get<int>(),
                 image_size["height"].template get<int>());

    RotatedDetectionResult detection_result;
    detection_result.objects = input_detections;
    detection_result.maps.push_back(
        {cv::Mat::ones(cv::Size(7, 7), CV_8UC1), roi, test_obj.label});

    return mediapipe::MakePacket<RotatedDetectionResult>(detection_result);
  }

  throw std::runtime_error("No packet creation defined.");
}

inference::ModelInferRequest build_request(std::string file_path,
                                           bool include_xai) {
  auto request = inference::ModelInferRequest();
  std::ifstream is(file_path);
  std::stringstream ss;
  ss << is.rdbuf();

  request.mutable_raw_input_contents()->Add(std::move(ss.str()));
  auto param = inference::InferParameter();
  param.set_bool_param(include_xai);
  (*request.mutable_parameters())["include_xai"] = param;
  return request;
}

TEST_P(CalculatorParameterizedTest, SerializationTests) {
  auto& type = GetParam().type;
  auto& input = GetParam().input;
  auto expected = GetParam().output;
  bool include_xai = GetParam().include_xai;
  if (!include_xai) {
    expected.erase("maps");
  }
  mediapipe::Packet packet = build_result_packet(input, type);
  auto request = build_request("/data/cattle.jpg", include_xai);
  mediapipe::Packet request_packet =
      mediapipe::MakePacket<const KFSRequest*>(&request);

  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config(type);
  mediapipe::tool::AddVectorSink("output", &graph_config, &output_packets);

  mediapipe::CalculatorGraph graph(graph_config);

  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", packet.At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "request", request_packet.At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  auto& response = output_packets[0].Get<KFSResponse*>();
  auto& actual_string = response->parameters().at("predictions").string_param();

  nlohmann::json actual = nlohmann::json::parse(actual_string);

  if (actual != expected) {
    auto diff = nlohmann::json::diff(expected, actual);
    for (auto& change : diff) {
      std::cout << change << std::endl;
    }
  }
  ASSERT_EQ(actual, expected);
}

std::string PrintToString(const TestData& test) { return test.test_name; }

struct PrintToStringParamName {
  template <class TestData>
  std::string operator()(const ::testing::TestParamInfo<TestData>& info) const {
    return PrintToString(info.param);
  }
};

INSTANTIATE_TEST_SUITE_P(
    SerializationCalculatorTestsWithXai, CalculatorParameterizedTest,
    testing::ValuesIn(GetTestsWithXai("serialization/tests.json")),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    SerializationCalculatorTestsWithoutXai, CalculatorParameterizedTest,
    testing::ValuesIn(GetTestsWithoutXai("serialization/tests.json")),
    PrintToStringParamName());

}  // namespace mediapipe
