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
//
// Reads frames from a video file with OpenCV, runs image classification for
// every frame through a MediaPipe graph built from OpenVINOSessionCalculator +
// OpenVINOInferenceCalculator, and prints the resulting tensor.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "openvino/openvino.hpp"

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Text format CalculatorGraphConfig proto with the OpenVINO nodes.");
ABSL_FLAG(std::string, input_video_path, "",
          "Video file read with cv::VideoCapture.");
ABSL_FLAG(int, input_width, 224, "Model input width.");
ABSL_FLAG(int, input_height, 224, "Model input height.");
ABSL_FLAG(bool, input_nhwc, false,
          "Build the input tensor as NHWC instead of NCHW.");
ABSL_FLAG(bool, swap_rb, true,
          "Convert the OpenCV BGR frame to RGB before inference.");
ABSL_FLAG(double, scale, 1.0,
          "Multiply every pixel value by this factor (e.g. 0.00392156862 to "
          "normalize to [0,1]).");
ABSL_FLAG(int, max_frames, 0, "Stop after N frames. 0 means the whole video.");
ABSL_FLAG(int, top_k, 5, "How many top scoring classes to print per frame.");
ABSL_FLAG(int, num_sources, 1,
          "Number of producer threads and graphs sharing one model session.");
ABSL_FLAG(bool, print_tensors, true, "Print tensors for each completed frame.");

namespace {

constexpr char kInputStream[] = "input_tensors";
constexpr char kOutputStream[] = "output_tensors";

// Converts a BGR cv::Mat into an fp32 ov::Tensor of shape
// {1, 3, h, w} (NCHW) or {1, h, w, 3} (NHWC).
ov::Tensor FrameToTensor(const cv::Mat& frame, int width, int height,
                         bool nhwc, bool swap_rb, double scale) {
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(width, height));
  if (swap_rb) {
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  }
  resized.convertTo(resized, CV_32FC3, scale);

  const ov::Shape shape = nhwc ? ov::Shape{1, static_cast<size_t>(height),
                                           static_cast<size_t>(width), 3}
                               : ov::Shape{1, 3, static_cast<size_t>(height),
                                           static_cast<size_t>(width)};
  ov::Tensor tensor(ov::element::f32, shape);
  float* dst = tensor.data<float>();
  if (nhwc) {
    std::copy_n(reinterpret_cast<const float*>(resized.data),
                tensor.get_size(), dst);
  } else {
    // Split the interleaved HWC buffer into planar CHW.
    const int plane = width * height;
    std::vector<cv::Mat> planes;
    for (int c = 0; c < 3; ++c) {
      planes.emplace_back(height, width, CV_32FC1, dst + c * plane);
    }
    cv::split(resized, planes);
  }
  return tensor;
}

void PrintTensor(int frame_index, const std::vector<ov::Tensor>& tensors,
                 int top_k) {
  for (size_t t = 0; t < tensors.size(); ++t) {
    const ov::Tensor& tensor = tensors[t];
    std::cout << "frame " << frame_index << " output[" << t
              << "] shape=" << tensor.get_shape()
              << " type=" << tensor.get_element_type() << "\n";
    if (tensor.get_element_type() != ov::element::f32) {
      continue;
    }
    const float* data = tensor.data<const float>();
    const size_t size = tensor.get_size();

    std::cout << "  values:";
    for (size_t i = 0; i < std::min<size_t>(size, 16); ++i) {
      std::cout << " " << data[i];
    }
    if (size > 16) std::cout << " ...";
    std::cout << "\n";

    if (top_k > 0) {
      std::vector<size_t> idx(size);
      std::iota(idx.begin(), idx.end(), 0);
      const size_t k = std::min<size_t>(size, static_cast<size_t>(top_k));
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [data](size_t a, size_t b) {
                          return data[a] > data[b];
                        });
      std::cout << "  top" << k << ":";
      for (size_t i = 0; i < k; ++i) {
        std::cout << " " << idx[i] << "=" << data[idx[i]];
      }
      std::cout << "\n";
    }
  }
  std::cout.flush();
}

absl::Status RunSource(mediapipe::CalculatorGraph& graph,
             mediapipe::OutputStreamPoller& poller, int source_index,
             int& completed_frames, std::mutex& output_mutex) {
  cv::VideoCapture capture(absl::GetFlag(FLAGS_input_video_path));
  RET_CHECK(capture.isOpened())
      << "Cannot open video " << absl::GetFlag(FLAGS_input_video_path);

  const int max_frames = absl::GetFlag(FLAGS_max_frames);
  int frame_index = 0;
  cv::Mat frame;
  while (capture.read(frame) && !frame.empty()) {
    if (max_frames > 0 && frame_index >= max_frames) break;

    ov::Tensor tensor = FrameToTensor(
        frame, absl::GetFlag(FLAGS_input_width),
        absl::GetFlag(FLAGS_input_height), absl::GetFlag(FLAGS_input_nhwc),
        absl::GetFlag(FLAGS_swap_rb), absl::GetFlag(FLAGS_scale));

    auto packet_data = std::make_unique<std::vector<ov::Tensor>>();
    packet_data->push_back(std::move(tensor));
    ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(packet_data.release())
                          .At(mediapipe::Timestamp(frame_index))));

    mediapipe::Packet output_packet;
    RET_CHECK(poller.Next(&output_packet)) << "Graph produced no output";
    RET_CHECK(!output_packet.Get<std::vector<ov::Tensor>>().empty())
        << "Graph produced no tensors";
    if (absl::GetFlag(FLAGS_print_tensors)) {
      std::lock_guard<std::mutex> lock(output_mutex);
      if (absl::GetFlag(FLAGS_num_sources) > 1) {
        std::cout << "source " << source_index << "\n";
      }
      PrintTensor(frame_index, output_packet.Get<std::vector<ov::Tensor>>(),
                  absl::GetFlag(FLAGS_top_k));
    }
    ++frame_index;
    completed_frames = frame_index;
  }
  capture.release();

  ABSL_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

absl::Status RunClassification() {
  RET_CHECK(!absl::GetFlag(FLAGS_calculator_graph_config_file).empty())
      << "--calculator_graph_config_file is required";
  RET_CHECK(!absl::GetFlag(FLAGS_input_video_path).empty())
      << "--input_video_path is required";
  const int num_sources = absl::GetFlag(FLAGS_num_sources);
  RET_CHECK_GT(num_sources, 0) << "--num_sources must be positive";

  std::string config_contents;
  ABSL_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file), &config_contents));
  const auto config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          config_contents);

  std::vector<std::unique_ptr<mediapipe::CalculatorGraph>> graphs;
  std::vector<mediapipe::OutputStreamPoller> pollers;
  for (int source = 0; source < num_sources; ++source) {
    auto graph = std::make_unique<mediapipe::CalculatorGraph>();
    ABSL_RETURN_IF_ERROR(graph->Initialize(config));
    ABSL_ASSIGN_OR_RETURN(auto poller,
                          graph->AddOutputStreamPoller(kOutputStream));
    ABSL_RETURN_IF_ERROR(graph->StartRun({}));
    graphs.push_back(std::move(graph));
    pollers.push_back(std::move(poller));
  }

  std::mutex output_mutex;
  std::vector<std::thread> producers;
  std::vector<absl::Status> statuses(num_sources);
  std::vector<int> completed_frames(num_sources, 0);
  const auto start = std::chrono::steady_clock::now();
  for (int source = 0; source < num_sources; ++source) {
    producers.emplace_back([&, source]() {
      try {
        statuses[source] = RunSource(*graphs[source], pollers[source], source,
                                     completed_frames[source], output_mutex);
      } catch (const std::exception& error) {
        statuses[source] = absl::InternalError(error.what());
      }
      if (!statuses[source].ok()) {
        graphs[source]->Cancel();
        graphs[source]->WaitUntilDone().IgnoreError();
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  int64_t total_frames = 0;
  for (int source = 0; source < num_sources; ++source) {
    std::cout << "source " << source << " frames=" << completed_frames[source]
              << "\n";
    total_frames += completed_frames[source];
    ABSL_RETURN_IF_ERROR(statuses[source]);
  }
  std::cout << "benchmark sources=" << num_sources << " frames=" << total_frames
            << " seconds=" << seconds
            << " aggregate_fps=" << total_frames / seconds
            << " fps_per_source=" << total_frames / seconds / num_sources
            << "\n";
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = RunClassification();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed: " << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
