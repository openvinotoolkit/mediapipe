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
#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <models/results.h>

#include <vector>

struct Label {
  std::string label_id;
  std::string label;
};

struct SaliencyMap {
  cv::Mat image;
  cv::Rect roi;
  Label label;
};

struct GetiDetectedObject {
  Label label;
  cv::Rect shape;
  float confidence;
};

struct GetiDetectionResult {
  cv::Size image_size;
  std::vector<GetiDetectedObject> objects;
  std::vector<SaliencyMap> maps;
};

struct GetiClassification {
  Label label;
  float score;
  cv::Rect roi;
};

struct GetiClassificationResult {
  std::vector<GetiClassification> predictions;
  std::vector<SaliencyMap> maps;
};
struct RotatedDetectedObject {
  Label label;
  float confidence;
  cv::RotatedRect rotatedRectangle;
};

struct RotatedDetectionResult {
  std::vector<RotatedDetectedObject> objects;
  std::vector<SaliencyMap> maps;
  ov::Tensor feature_vector;
};

struct DetectionClassification {
  GetiDetectedObject detection;
  GetiClassificationResult classifications;
};

struct DetectionClassificationResult {
  std::vector<DetectionClassification> predictions;
};

struct GetiContour {
  Label label;
  float probability;
  std::vector<cv::Point> shape;
};

struct SegmentationResult {
  std::vector<GetiContour> contours;
  std::vector<SaliencyMap> maps;
  ov::Tensor feature_vector;
};

struct GetiAnomalyResult {
  std::vector<GetiDetectedObject> detections;
  std::vector<GetiContour> segmentations;
  std::vector<SaliencyMap> maps;
};

struct DetectionSegmentation {
  GetiDetectedObject detection_result;
  SegmentationResult segmentation_result;
};

struct DetectionSegmentationResult {
  GetiDetectionResult detection;
  std::vector<DetectionSegmentation> segmentations;
};

#endif  // DATA_STRUCTURES_H
