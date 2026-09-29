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
// Runs GStreamer VA-decoded video frames through a MediaPipe OpenVINO
// classification graph and prints the resulting tensors.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "google/protobuf/any.pb.h"
#include "mediapipe/calculators/openvino/gstreamer_remote_tensor_calculator.pb.h"
#include "mediapipe/calculators/openvino/openvino_session_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "openvino/openvino.hpp"

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Text format CalculatorGraphConfig proto with the OpenVINO nodes.");
ABSL_FLAG(std::string, input_video_path, "",
          "Video file or URI decoded by GStreamer VA-API.");
ABSL_FLAG(std::string, model_path, "",
          "OpenVINO ResNet model XML path.");
ABSL_FLAG(std::string, va_device_path, "/dev/dri/renderD128",
          "Linux VA-API render node shared with OpenVINO GPU.");
ABSL_FLAG(int, max_frames, 0, "Stop after N frames. 0 means the whole video.");
ABSL_FLAG(int, top_k, 5, "How many top scoring classes to print per frame.");

namespace {

constexpr char kOutputStream[] = "output_tensors";

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

absl::Status RunClassification() {
  RET_CHECK(!absl::GetFlag(FLAGS_calculator_graph_config_file).empty())
      << "--calculator_graph_config_file is required";
  RET_CHECK(!absl::GetFlag(FLAGS_input_video_path).empty())
      << "--input_video_path is required";
  RET_CHECK(!absl::GetFlag(FLAGS_model_path).empty())
      << "--model_path is required";
  RET_CHECK_GE(absl::GetFlag(FLAGS_max_frames), 0)
      << "--max_frames cannot be negative";

  std::string config_contents;
  ABSL_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file), &config_contents));
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          config_contents);

  bool session_options_updated = false;
  bool source_options_updated = false;
  for (auto& node : *config.mutable_node()) {
    for (int i = 0; i < node.node_options_size(); ++i) {
      google::protobuf::Any* any = node.mutable_node_options(i);
      if (any->Is<mediapipe::OpenVINOSessionCalculatorOptions>()) {
        mediapipe::OpenVINOSessionCalculatorOptions options;
        RET_CHECK(any->UnpackTo(&options))
            << "Cannot unpack OpenVINOSessionCalculatorOptions";
        options.set_model_path(absl::GetFlag(FLAGS_model_path));
        options.set_va_device_path(absl::GetFlag(FLAGS_va_device_path));
        any->PackFrom(options);
        session_options_updated = true;
      } else if (any->Is<mediapipe::GStreamerRemoteTensorCalculatorOptions>()) {
        mediapipe::GStreamerRemoteTensorCalculatorOptions options;
        RET_CHECK(any->UnpackTo(&options))
            << "Cannot unpack GStreamerRemoteTensorCalculatorOptions";
        options.set_source(absl::GetFlag(FLAGS_input_video_path));
        options.set_max_frames(absl::GetFlag(FLAGS_max_frames));
        any->PackFrom(options);
        source_options_updated = true;
      }
    }
  }
  RET_CHECK(session_options_updated)
      << "Graph must contain OpenVINOSessionCalculatorOptions";
  RET_CHECK(source_options_updated)
      << "Graph must contain GStreamerRemoteTensorCalculatorOptions";

  mediapipe::CalculatorGraph graph;
  ABSL_RETURN_IF_ERROR(graph.Initialize(config));
  ABSL_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                        graph.AddOutputStreamPoller(kOutputStream));
  ABSL_RETURN_IF_ERROR(graph.StartRun({}));

  int frame_index = 0;
  mediapipe::Packet output_packet;
  while (poller.Next(&output_packet)) {
    PrintTensor(frame_index, output_packet.Get<std::vector<ov::Tensor>>(),
                absl::GetFlag(FLAGS_top_k));
    ++frame_index;
  }
  return graph.WaitUntilDone();
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
