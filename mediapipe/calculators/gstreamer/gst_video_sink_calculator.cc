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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>
#include <variant>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/gstreamer/gst_video_frame_packet.h"
#include "mediapipe/calculators/gstreamer/mpi/intel_mpi.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {

constexpr char kFrameTag[] = "FRAME";
constexpr char kOutputPathTag[] = "OUTPUT_PATH";
constexpr char kFramerateTag[] = "FRAMERATE";
constexpr char kOutputModeTag[] = "OUTPUT_MODE";

}  // namespace

// Encodes GstVideoFramePacket frames to H.264. GPU surfaces are encoded in
// place (VA on Linux, D3D11 on Windows); host NV12 frames use a CPU encoder.
//
// Input stream:
//   FRAME       - GstVideoFramePacket.
// Input side packets:
//   OUTPUT_PATH - std::string: MP4 file path (file) or rtsp:// URL (rtsp).
//   FRAMERATE   - int, encode framerate.
//   OUTPUT_MODE - std::string "file" (default), "rtsp" or "preview"
//                 (Windows only), or bool (true = rtsp). Optional.
//
//   node {
//     calculator: "GstVideoSinkCalculator"
//     input_stream: "FRAME:frame"
//     input_side_packet: "OUTPUT_PATH:output_path"
//     input_side_packet: "FRAMERATE:framerate"
//   }
class GstVideoSinkCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kFrameTag).Set<GstVideoFramePacket>();
    if (cc->InputSidePackets().HasTag(kOutputPathTag)) {
      cc->InputSidePackets().Tag(kOutputPathTag).Set<std::string>();
    }
    cc->InputSidePackets().Tag(kFramerateTag).Set<int>();
    if (cc->InputSidePackets().HasTag(kOutputModeTag)) {
      cc->InputSidePackets().Tag(kOutputModeTag).SetOneOf<std::string, bool>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    if (cc->InputSidePackets().HasTag(kOutputPathTag)) {
      output_path_ =
          cc->InputSidePackets().Tag(kOutputPathTag).Get<std::string>();
    }
    framerate_ = cc->InputSidePackets().Tag(kFramerateTag).Get<int>();
    if (cc->InputSidePackets().HasTag(kOutputModeTag)) {
      const Packet& mode = cc->InputSidePackets().Tag(kOutputModeTag);
      if (mode.ValidateAsType<bool>().ok()) {
        output_mode_ = mode.Get<bool>() ? IMP_OUTPUT_RTSP : IMP_OUTPUT_FILE;
      } else {
        std::string name = mode.Get<std::string>();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (name == "file") {
          output_mode_ = IMP_OUTPUT_FILE;
        } else if (name == "rtsp") {
          output_mode_ = IMP_OUTPUT_RTSP;
        } else if (name == "preview") {
          output_mode_ = IMP_OUTPUT_PREVIEW;
        } else {
          return absl::InvalidArgumentError(
              "GstVideoSinkCalculator: OUTPUT_MODE must be 'file', 'rtsp' or "
              "'preview'");
        }
      }
    }
#ifndef _WIN32
    RET_CHECK(output_mode_ != IMP_OUTPUT_PREVIEW)
        << "GstVideoSinkCalculator: preview output requires Windows";
#endif
    RET_CHECK(output_mode_ == IMP_OUTPUT_PREVIEW || !output_path_.empty())
        << "GstVideoSinkCalculator: empty OUTPUT_PATH";
    RET_CHECK_GT(framerate_, 0) << "GstVideoSinkCalculator: FRAMERATE";
    if (imp_context_create(&ctx_, nullptr) != IMP_OK) {
      return absl::InternalError("imp_context_create failed");
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& frame = cc->Inputs().Tag(kFrameTag).Get<GstVideoFramePacket>();
    imp_tensor_t* tensor = frame.owner.get();
    RET_CHECK(tensor != nullptr) << "frame packet has no native tensor owner";

    // Frame dimensions are known only once the first frame arrives.
    if (!encoder_) {
      RET_CHECK_GT(frame.width, 0);
      RET_CHECK_GT(frame.height, 0);
      const bool native_surface =
          !std::holds_alternative<CpuNv12Frame>(frame.resource);
      imp_video_encode_opts_t options{};
      options.codec = "h264";
      options.framerate = static_cast<uint32_t>(framerate_);
      options.encode_device = native_surface ? "GPU" : nullptr;
      options.output_path = output_path_.c_str();
      options.output_mode = output_mode_;
      if (imp_video_encoder_create(&encoder_,
                                   static_cast<uint32_t>(frame.width),
                                   static_cast<uint32_t>(frame.height), ctx_,
                                   &options) != IMP_OK) {
        return absl::InternalError(
            absl::StrCat("imp_video_encoder_create failed: ",
                         imp_context_get_error(ctx_)));
      }
    }

    if (imp_video_encoder_write(encoder_, tensor) != IMP_OK) {
      return absl::InternalError(absl::StrCat(
          "imp_video_encoder_write failed: ", imp_context_get_error(ctx_)));
    }
    ++frames_written_;
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    if (encoder_) {
      imp_video_encoder_close(encoder_);
      encoder_ = nullptr;
    }
    if (ctx_) {
      imp_context_destroy(ctx_);
      ctx_ = nullptr;
    }
    RET_CHECK_GT(frames_written_, 0)
        << "GstVideoSinkCalculator: no frames were encoded";
    if (output_mode_ != IMP_OUTPUT_FILE) return absl::OkStatus();
    std::ifstream out(output_path_, std::ios::binary | std::ios::ate);
    RET_CHECK(out.good() && out.tellg() > 0)
        << "GstVideoSinkCalculator: encoder did not finalize " << output_path_;
    return absl::OkStatus();
  }

 private:
  std::string output_path_;
  int framerate_ = 0;
  imp_output_mode_t output_mode_ = IMP_OUTPUT_FILE;
  int64_t frames_written_ = 0;

  imp_context_t* ctx_ = nullptr;
  imp_video_encoder_t* encoder_ = nullptr;
};

REGISTER_CALCULATOR(GstVideoSinkCalculator);

}  // namespace mediapipe
