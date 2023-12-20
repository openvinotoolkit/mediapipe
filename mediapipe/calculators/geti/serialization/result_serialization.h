#ifndef RESULT_SERIALIZATION_H_
#define RESULT_SERIALIZATION_H_

#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "nlohmann/json.hpp"
#include "third_party/cpp-base64/base64.h"
#include "mediapipe/calculators/geti/utils/data_structures.h"

namespace geti {
static inline std::string base64_encode_mat(cv::Mat image) {
  std::vector<uchar> buf;
  if (!image.empty()) cv::imencode(".jpg", image, buf);
  auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
  return base64_encode(enc_msg, buf.size());
}

static inline nlohmann::json add_maps(const std::vector<SaliencyMap> &maps) {
  auto json_maps = nlohmann::json::array();
  for (auto &map : maps) {
    if (!map.image.empty()) {
      json_maps.push_back({{"label_id", map.label.label_id},
                           {"label_name", map.label.label},
                           {"data", base64_encode_mat(map.image)}});
    }
  }
  return json_maps;
}

static inline nlohmann::json serialize(const GetiDetectionResult &result,
                                       bool include_xai) {
  const auto &objects = result.objects;

  nlohmann::json predictions;
  for (auto &obj : objects) {
    predictions.push_back({
        {"labels", nlohmann::json::array({{{"name", obj.label.label},
                                           {"id", obj.label.label_id},
                                           {"probability", obj.confidence}}})},
        {"shape",
         {
             {"type", "RECTANGLE"},
             {"height", obj.shape.height},
             {"width", obj.shape.width},
             {"x", obj.shape.x},
             {"y", obj.shape.y},
         }},

    });
  }

  nlohmann::json data = {{"predictions", predictions}};

  if (include_xai) {
    data["maps"] = add_maps(result.maps);
  }

  return data;
}

static inline nlohmann::json serialize(const RotatedDetectionResult &result,
                                       bool include_xai) {
  const auto &objects = result.objects;

  nlohmann::json predictions;
  for (auto &obj : objects) {
    predictions.push_back({
        {"labels", nlohmann::json::array({{{"name", obj.label.label},
                                           {"id", obj.label.label_id},
                                           {"probability", obj.confidence}}})},
        {"shape",
         {
             {"type", "ROTATED_RECTANGLE"},
             {"height", obj.rotatedRectangle.size.height},
             {"width", obj.rotatedRectangle.size.width},
             {"x", obj.rotatedRectangle.center.x},
             {"y", obj.rotatedRectangle.center.y},
             {"angle", obj.rotatedRectangle.angle},
         }},

    });
  }

  nlohmann::json data = {{"predictions", predictions}};
  if (include_xai) {
    data["maps"] = add_maps(result.maps);
  }

  return data;
}

static inline nlohmann::json serialize(const DetectionClassification &result,
                                       bool include_xai) {
  nlohmann::json labels;
  const auto &detection = result.detection;
  labels.push_back({{"name", detection.label.label},
                    {"id", detection.label.label_id},
                    {"probability", detection.confidence}});

  for (auto &classification : result.classifications.predictions) {
    labels.push_back({{"name", classification.label.label},
                      {"id", classification.label.label_id},
                      {"probability", classification.score}});
  }
  return {
      {"labels", labels},
      {"shape",
       {
           {"type", "RECTANGLE"},
           {"height", detection.shape.height},
           {"width", detection.shape.width},
           {"x", detection.shape.x},
           {"y", detection.shape.y},
       }},
  };
}

static inline nlohmann::json serialize(
    const DetectionClassificationResult &result, bool include_xai) {
  nlohmann::json predictions;
  for (const auto &prediction : result.predictions) {
    predictions.push_back(serialize(prediction, include_xai));
  }
  nlohmann::json data = {{"predictions", predictions}};
  // Disabled for now...
  // if (include_xai) {
  //  data["maps"] = add_maps(result.maps);
  //}
  return data;
}

static inline nlohmann::json serialize(const GetiClassificationResult &result,
                                       bool include_xai) {
  const auto &objects = result.predictions;

  nlohmann::json predictions;
  for (auto &obj : objects) {
    predictions.push_back({
        {"labels", nlohmann::json::array({{{"name", obj.label.label},
                                           {"id", obj.label.label_id},
                                           {"probability", obj.score}}})},
        {"shape",
         {
             {"type", "RECTANGLE"},
             {"height", obj.roi.height},
             {"width", obj.roi.width},
             {"x", obj.roi.x},
             {"y", obj.roi.y},
         }},

    });
  }

  nlohmann::json data = {{"predictions", predictions}};
  if (include_xai) {
    data["maps"] = add_maps(result.maps);
  }
  return data;
}

static inline nlohmann::json serialize(const SegmentationResult &result,
                                       bool include_xai) {
  const auto &objects = result.contours;

  nlohmann::json predictions;
  for (auto &obj : objects) {
    nlohmann::json points_json;
    for (auto &coord : obj.shape) {
      points_json.push_back({{"x", coord.x}, {"y", coord.y}});
    }
    predictions.push_back(
        {{"labels",
          nlohmann::json::array({{{"name", obj.label.label},
                                  {"id", obj.label.label_id},
                                  {"probability", obj.probability}}})},
         {"shape", {{"points", points_json}, {"type", "POLYGON"}}}});
  }

  nlohmann::json data = {{"predictions", predictions}};
  if (include_xai) {
    data["maps"] = add_maps(result.maps);
  }

  return data;
}

static inline nlohmann::json serialize(const DetectionSegmentation &result,
                                       bool include_xai) {
  return serialize(result.segmentation_result, include_xai);
}

static inline nlohmann::json serialize(
    const DetectionSegmentationResult &result, bool include_xai) {
  auto data = serialize(result.detection, include_xai);

  for (const auto &prediction : result.segmentations) {
    const auto segmentation_predictions =
        serialize(prediction, false)["predictions"];
    data["predictions"].insert(data["predictions"].end(),
                               segmentation_predictions.begin(),
                               segmentation_predictions.end());
  }

  return data;
}

static inline nlohmann::json serialize(const GetiAnomalyResult &result,
                                       bool include_xai) {
  nlohmann::json predictions;
  for (auto &obj : result.detections) {
    predictions.push_back({
        {"labels", nlohmann::json::array({{{"name", obj.label.label},
                                           {"id", obj.label.label_id},
                                           {"probability", obj.confidence}}})},
        {"shape",
         {
             {"type", "RECTANGLE"},
             {"height", obj.shape.height},
             {"width", obj.shape.width},
             {"x", obj.shape.x},
             {"y", obj.shape.y},
         }},
    });
  }
  for (auto &obj : result.segmentations) {
    nlohmann::json points_json;
    for (auto &coord : obj.shape) {
      points_json.push_back({{"x", coord.x}, {"y", coord.y}});
    }
    predictions.push_back(
        {{"labels",
          nlohmann::json::array({{{"name", obj.label.label},
                                  {"id", obj.label.label_id},
                                  {"probability", obj.probability}}})},
         {"shape", {{"points", points_json}, {"type", "POLYGON"}}}});
  }

  nlohmann::json data = {{"predictions", predictions}};
  if (include_xai) {
    data["maps"] = add_maps(result.maps);
  }
  return data;
}
}  // namespace geti

#endif  // RESULT_SERIALIZATION_H_
