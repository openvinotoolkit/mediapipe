// Copyright 2026 Intel Corporation
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

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/gstreamer/gst_detection_box.h"
#include "mediapipe/calculators/gstreamer/gst_video_frame_packet.h"
#include "mediapipe/calculators/gstreamer/mpi/intel_mpi.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset8.hpp"

namespace mediapipe {
namespace {

constexpr int kModelWidth = 64;
constexpr int kModelHeight = 64;
constexpr int kVideoWidth = 128;
constexpr int kVideoHeight = 96;
constexpr int kFrames = 4;

#ifdef _WIN32
constexpr char kRemoteContext[] = "D3D11";
#else
constexpr char kRemoteContext[] = "VA";
#endif

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return absl::StrCat(dir != nullptr ? dir : ".", "/", name);
}

// Returns the number of decodable video frames, or -1 when ffprobe fails.
int CountVideoFrames(const std::string& path) {
  const std::string command = absl::StrCat(
      "ffprobe -v error -count_frames -select_streams v:0 -show_entries "
      "stream=nb_read_frames -of csv=p=0 \"",
      path, "\"");
#ifdef _WIN32
  FILE* pipe = _popen(command.c_str(), "r");
#else
  FILE* pipe = popen(command.c_str(), "r");
#endif
  if (pipe == nullptr) return -1;
  int frames = -1;
  if (std::fscanf(pipe, "%d", &frames) != 1) frames = -1;
#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  return frames;
}

class GstVideoGpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!imp_video_va_available() && !imp_video_d3d11_available()) {
      skip_reason_ = new std::string("no GStreamer VA/D3D11 decode path");
      return;
    }
    try {
      if (ov::Core().get_property("GPU", ov::device::full_name).empty()) {
        skip_reason_ = new std::string("no OpenVINO GPU device");
        return;
      }
    } catch (const std::exception& e) {
      skip_reason_ = new std::string(e.what());
      return;
    }
    video_path_ = new std::string(TempPath("input.mp4"));
    const std::string command = absl::StrCat(
        "ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc=size=",
        kVideoWidth, "x", kVideoHeight, ":rate=10 -frames:v ", kFrames + 2,
        " -c:v libx264 -pix_fmt yuv420p \"", *video_path_, "\"");
    if (std::system(command.c_str()) != 0) {
      skip_reason_ = new std::string("ffmpeg cannot create the test video");
      return;
    }
    model_path_ = new std::string(TempPath("channel_mean.xml"));
    WriteChannelMeanModel(*model_path_);
  }

  void SetUp() override {
    if (skip_reason_ != nullptr) GTEST_SKIP() << *skip_reason_;
  }

  // f32 [1,3,H,W] -> per-channel mean [1,3,1,1].
  static void WriteChannelMeanModel(const std::string& path) {
    auto input = std::make_shared<ov::opset8::Parameter>(
        ov::element::f32,
        ov::Shape{1, 3, kModelHeight, kModelWidth});
    input->set_friendly_name("data");
    input->get_output_tensor(0).set_names({"data"});
    auto axes = ov::opset8::Constant::create(ov::element::i64, {2}, {2, 3});
    auto mean = std::make_shared<ov::opset8::ReduceMean>(input, axes, true);
    mean->get_output_tensor(0).set_names({"mean"});
    auto result = std::make_shared<ov::opset8::Result>(mean);
    ov::serialize(std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{input}),
                  path);
  }

  static std::string SessionNode() {
    return absl::StrReplaceAll(R"pb(
      node {
        calculator: "OpenVINOSessionCalculator"
        output_side_packet: "SESSION:session"
        node_options: {
          [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
            model_path: "$MODEL"
            device: "GPU"
            remote_context: $CONTEXT
            layout: "NHWC:NCHW"
            color_format: "NV12_2:BGR"
            precision: "u8:fp32"
            num_infer_requests: 2
          }
        }
      }
    )pb",
                               {{"$MODEL", *model_path_},
                                {"$CONTEXT", kRemoteContext}});
  }

  static std::map<std::string, Packet> SourceSidePackets() {
    return {{"video_path", MakePacket<std::string>(*video_path_)},
            {"width", MakePacket<int>(kModelWidth)},
            {"height", MakePacket<int>(kModelHeight)},
            {"max_frames", MakePacket<int>(kFrames)}};
  }

  static std::string* skip_reason_;
  static std::string* video_path_;
  static std::string* model_path_;
};

std::string* GstVideoGpuTest::skip_reason_ = nullptr;
std::string* GstVideoGpuTest::video_path_ = nullptr;
std::string* GstVideoGpuTest::model_path_ = nullptr;

TEST_F(GstVideoGpuTest, InfersOnDecodedSurfaces) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrCat(
      SessionNode(), R"pb(
        input_side_packet: "video_path"
        input_side_packet: "width"
        input_side_packet: "height"
        input_side_packet: "max_frames"
        node {
          calculator: "GstVideoSourceCalculator"
          input_side_packet: "SESSION:session"
          input_side_packet: "VIDEO_PATH:video_path"
          input_side_packet: "WIDTH:width"
          input_side_packet: "HEIGHT:height"
          input_side_packet: "MAX_FRAMES:max_frames"
          output_stream: "FRAME:frame"
        }
        node {
          calculator: "OpenVINOInferenceCalculator"
          input_side_packet: "SESSION:session"
          input_stream: "VIDEO_FRAME:frame"
          output_stream: "TENSORS:output"
        }
      )pb"));
  std::vector<Packet> outputs;
  tool::AddVectorSink("output", &config, &outputs);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun(SourceSidePackets()));
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(outputs.size(), kFrames);
  for (const Packet& packet : outputs) {
    const auto& tensors = packet.Get<std::vector<ov::Tensor>>();
    ASSERT_EQ(tensors.size(), 1);
    ASSERT_EQ(tensors[0].get_shape(), (ov::Shape{1, 3, 1, 1}));
    const float* mean = tensors[0].data<const float>();
    for (int channel = 0; channel < 3; ++channel) {
      EXPECT_GT(mean[channel], 0.0f);
      EXPECT_LT(mean[channel], 255.0f);
    }
  }
}

class GstVideoGpuOverlayTest : public GstVideoGpuTest,
                               public ::testing::WithParamInterface<bool> {};

TEST_P(GstVideoGpuOverlayTest, DrawsAndEncodesFullFrames) {
  const bool native_full_frame = GetParam();
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrCat(
      SessionNode(), R"pb(
        input_stream: "boxes"
        input_side_packet: "video_path"
        input_side_packet: "width"
        input_side_packet: "height"
        input_side_packet: "max_frames"
        input_side_packet: "full_native_surface"
        input_side_packet: "full_width"
        input_side_packet: "full_height"
        input_side_packet: "output_path"
        input_side_packet: "framerate"
        node {
          calculator: "GstVideoSourceCalculator"
          input_side_packet: "SESSION:session"
          input_side_packet: "VIDEO_PATH:video_path"
          input_side_packet: "WIDTH:width"
          input_side_packet: "HEIGHT:height"
          input_side_packet: "MAX_FRAMES:max_frames"
          input_side_packet: "FULL_NATIVE_SURFACE:full_native_surface"
          input_side_packet: "FULL_WIDTH:full_width"
          input_side_packet: "FULL_HEIGHT:full_height"
          output_stream: "FRAME_INFER:frame"
          output_stream: "FRAME_FULL:frame_full"
        }
        node {
          calculator: "GstDetectionOverlayCalculator"
          input_stream: "FRAME_FULL:frame_full"
          input_stream: "DETECTION_BOXES:boxes"
          output_stream: "FRAME_OVERLAID:frame_overlaid"
        }
        node {
          calculator: "GstVideoSinkCalculator"
          input_stream: "FRAME:frame_overlaid"
          input_side_packet: "OUTPUT_PATH:output_path"
          input_side_packet: "FRAMERATE:framerate"
        }
      )pb"));
  const std::string output_path = TempPath(
      absl::StrCat("overlay_", native_full_frame ? "native" : "host", ".mp4"));
  std::map<std::string, Packet> side_packets = SourceSidePackets();
  side_packets["full_native_surface"] = MakePacket<bool>(native_full_frame);
  side_packets["full_width"] = MakePacket<int>(kVideoWidth);
  side_packets["full_height"] = MakePacket<int>(kVideoHeight);
  side_packets["output_path"] = MakePacket<std::string>(output_path);
  side_packets["framerate"] = MakePacket<int>(10);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun(side_packets));
  for (int frame = 0; frame < kFrames; ++frame) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "boxes", MakePacket<GstDetectionBoxes>(
                     GstDetectionBoxes{{0.9f, 10.0f, 10.0f, 60.0f, 50.0f}})
                     .At(Timestamp(frame))));
  }
  MP_ASSERT_OK(graph.CloseInputStream("boxes"));
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(CountVideoFrames(output_path), kFrames);
}

INSTANTIATE_TEST_SUITE_P(FullFrameMemory, GstVideoGpuOverlayTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Native" : "Host";
                         });

}  // namespace
}  // namespace mediapipe
