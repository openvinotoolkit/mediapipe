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
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/gstreamer/gst_video_frame_packet.h"
#include "mediapipe/calculators/gstreamer/mpi/intel_mpi.h"
#include "mediapipe/calculators/openvino/openvino_session.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/status_util.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

using ::mediapipe::openvino_calculators::OpenVINOSessionPtr;
using ::mediapipe::openvino_calculators::RemoteDeviceType;

constexpr char kSessionTag[] = "SESSION";
constexpr char kVideoPathTag[] = "VIDEO_PATH";
constexpr char kSourceModeTag[] = "SOURCE_MODE";
constexpr char kSourceUriTag[] = "SOURCE_URI";
constexpr char kCameraDeviceTag[] = "CAMERA_DEVICE";
constexpr char kWidthTag[] = "WIDTH";
constexpr char kHeightTag[] = "HEIGHT";
constexpr char kFullWidthTag[] = "FULL_WIDTH";
constexpr char kFullHeightTag[] = "FULL_HEIGHT";
constexpr char kFullNativeSurfaceTag[] = "FULL_NATIVE_SURFACE";
constexpr char kMaxFramesTag[] = "MAX_FRAMES";
constexpr char kFrameTag[] = "FRAME";
constexpr char kFrameInferTag[] = "FRAME_INFER";
constexpr char kFrameFullTag[] = "FRAME_FULL";
constexpr char kCompletionTag[] = "COMPLETION";

}  // namespace

// Decodes a video with GStreamer (through the Intel MPI library) and emits one
// GstVideoFramePacket per frame. Each packet owns its native NV12 frame: a VA
// surface on Linux or a D3D11 texture on Windows.
//
// Outputs (at least FRAME or FRAME_INFER):
//   FRAME       - model-resolution GPU surface.
//   FRAME_INFER - same as FRAME; name used by overlay graphs.
//   FRAME_FULL  - full-resolution frame for overlay/encode. Host NV12 by
//                 default, GPU surface with FULL_NATIVE_SURFACE=true. Decoded
//                 by a second stream on the same source, read in lockstep.
//   COMPLETION  - ov::Tensor emitted once MAX_FRAMES is reached.
//
// Input side packets:
//   SESSION             - OpenVINOSessionPtr (optional). When compiled with
//                         remote_context, GStreamer decodes on the session's
//                         VADisplay/ID3D11Device, so that frames can be
//                         consumed by OpenVINOInferenceCalculator VIDEO_FRAME.
//   VIDEO_PATH          - std::string, input video file.
//   SOURCE_MODE         - std::string (optional): "file" (default), "camera"
//                         or "rtsp".
//   SOURCE_URI          - std::string (optional): file path or RTSP URI,
//                         overrides VIDEO_PATH.
//   CAMERA_DEVICE       - std::string (optional) camera id for camera mode.
//   WIDTH, HEIGHT       - int, FRAME/FRAME_INFER resolution.
//   FULL_WIDTH/HEIGHT   - int (optional), FRAME_FULL resolution
//                         (0 = source resolution).
//   FULL_NATIVE_SURFACE - bool (optional).
//   MAX_FRAMES          - int or int64 (optional), 0 = whole stream.
//
//   node {
//     calculator: "GstVideoSourceCalculator"
//     input_side_packet: "SESSION:session"
//     input_side_packet: "VIDEO_PATH:video_path"
//     input_side_packet: "WIDTH:width"
//     input_side_packet: "HEIGHT:height"
//     output_stream: "FRAME:frame"
//   }
class GstVideoSourceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->InputSidePackets().HasTag(kSessionTag)) {
      cc->InputSidePackets().Tag(kSessionTag).Set<OpenVINOSessionPtr>();
    }
    for (const char* tag :
         {kVideoPathTag, kSourceModeTag, kSourceUriTag, kCameraDeviceTag}) {
      if (cc->InputSidePackets().HasTag(tag)) {
        cc->InputSidePackets().Tag(tag).Set<std::string>();
      }
    }
    RET_CHECK(cc->InputSidePackets().HasTag(kVideoPathTag) ||
              cc->InputSidePackets().HasTag(kSourceUriTag) ||
              cc->InputSidePackets().HasTag(kSourceModeTag))
        << "GstVideoSourceCalculator needs VIDEO_PATH, SOURCE_URI, or "
           "SOURCE_MODE";
    cc->InputSidePackets().Tag(kWidthTag).Set<int>();
    cc->InputSidePackets().Tag(kHeightTag).Set<int>();
    if (cc->InputSidePackets().HasTag(kFullWidthTag)) {
      cc->InputSidePackets().Tag(kFullWidthTag).Set<int>();
    }
    if (cc->InputSidePackets().HasTag(kFullHeightTag)) {
      cc->InputSidePackets().Tag(kFullHeightTag).Set<int>();
    }
    if (cc->InputSidePackets().HasTag(kFullNativeSurfaceTag)) {
      cc->InputSidePackets().Tag(kFullNativeSurfaceTag).Set<bool>();
    }
    if (cc->InputSidePackets().HasTag(kMaxFramesTag)) {
      cc->InputSidePackets().Tag(kMaxFramesTag).SetOneOf<int, int64_t>();
    }
    for (const char* tag : {kFrameTag, kFrameInferTag, kFrameFullTag}) {
      if (cc->Outputs().HasTag(tag)) {
        cc->Outputs().Tag(tag).Set<GstVideoFramePacket>();
      }
    }
    if (cc->Outputs().HasTag(kCompletionTag)) {
      cc->Outputs().Tag(kCompletionTag).Set<ov::Tensor>();
    }
    RET_CHECK(cc->Outputs().HasTag(kFrameTag) ||
              cc->Outputs().HasTag(kFrameInferTag))
        << "GstVideoSourceCalculator needs a FRAME or FRAME_INFER output";
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& side = cc->InputSidePackets();
    if (side.HasTag(kVideoPathTag)) {
      source_uri_ = side.Tag(kVideoPathTag).Get<std::string>();
    }
    if (side.HasTag(kSourceUriTag)) {
      source_uri_ = side.Tag(kSourceUriTag).Get<std::string>();
    }
    if (side.HasTag(kCameraDeviceTag)) {
      camera_device_ = side.Tag(kCameraDeviceTag).Get<std::string>();
    }
    if (side.HasTag(kSourceModeTag)) {
      std::string mode = side.Tag(kSourceModeTag).Get<std::string>();
      std::transform(mode.begin(), mode.end(), mode.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (mode == "file") {
        source_type_ = IMP_SOURCE_FILE;
      } else if (mode == "camera") {
        source_type_ = IMP_SOURCE_CAMERA;
      } else if (mode == "rtsp") {
        source_type_ = IMP_SOURCE_URL;
      } else {
        return absl::InvalidArgumentError(
            "GstVideoSourceCalculator: SOURCE_MODE must be 'file', 'camera', "
            "or 'rtsp'");
      }
    }
    if (source_type_ != IMP_SOURCE_CAMERA && source_uri_.empty()) {
      return absl::InvalidArgumentError(
          "GstVideoSourceCalculator: empty source URI");
    }
    width_ = side.Tag(kWidthTag).Get<int>();
    height_ = side.Tag(kHeightTag).Get<int>();
    if (side.HasTag(kFullWidthTag)) {
      full_width_ = side.Tag(kFullWidthTag).Get<int>();
    }
    if (side.HasTag(kFullHeightTag)) {
      full_height_ = side.Tag(kFullHeightTag).Get<int>();
    }
    if (side.HasTag(kFullNativeSurfaceTag)) {
      full_native_surface_ = side.Tag(kFullNativeSurfaceTag).Get<bool>();
    }
    if (side.HasTag(kMaxFramesTag)) {
      const Packet& limit = side.Tag(kMaxFramesTag);
      max_frames_ = limit.ValidateAsType<int64_t>().ok() ? limit.Get<int64_t>()
                                                         : limit.Get<int>();
    }

    emit_frame_ = cc->Outputs().HasTag(kFrameTag);
    emit_infer_ = cc->Outputs().HasTag(kFrameInferTag);
    emit_full_ = cc->Outputs().HasTag(kFrameFullTag);

    use_d3d11_ = imp_video_d3d11_available();
    if (!use_d3d11_ && !imp_video_va_available()) {
      return absl::UnavailableError(
          "GstVideoSourceCalculator: no GPU surface path (VA/D3D11) "
          "available");
    }
    ABSL_RETURN_IF_ERROR(ShareSessionDevice(cc));

#ifdef _WIN32
    if (use_d3d11_ && source_type_ == IMP_SOURCE_CAMERA && emit_full_) {
      imp_video_branch_t branches[2]{};
      branches[0].width = width_;
      branches[0].height = height_;
      branches[0].name = "inference";
      branches[1].width = full_width_;
      branches[1].height = full_height_;
      branches[1].name = "overlay";
      branches[1].output_memory = full_native_surface_
                                      ? IMP_VIDEO_MEMORY_DEFAULT
                                      : IMP_VIDEO_MEMORY_SYSTEM;
      ABSL_RETURN_IF_ERROR(OpenStream(&infer_ctx_, &infer_stream_, full_width_,
                                    full_height_, /*native_surface=*/true,
                                    &pending_infer_, IMP_MEM_D3D11_SURFACE,
                                    branches, 2));
      shared_camera_stream_ = true;
      if (imp_video_read_frame(&pending_full_, infer_stream_, 1) != IMP_OK ||
          !pending_full_) {
        return absl::InternalError("first camera overlay frame read failed");
      }
      return absl::OkStatus();
    }
#endif
    const imp_tensor_memory_type_t native_memory =
        use_d3d11_ ? IMP_MEM_D3D11_SURFACE : IMP_MEM_VA_SURFACE;
    if (emit_frame_ || emit_infer_) {
      ABSL_RETURN_IF_ERROR(OpenStream(&infer_ctx_, &infer_stream_, width_,
                                    height_, /*native_surface=*/true,
                                    &pending_infer_, native_memory));
    }
    if (emit_full_) {
      ABSL_RETURN_IF_ERROR(OpenStream(
          &full_ctx_, &full_stream_, full_width_, full_height_,
          full_native_surface_, &pending_full_,
          full_native_surface_ ? native_memory : IMP_MEM_SYSTEM));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (max_frames_ > 0 && frame_index_ >= max_frames_) {
      if (cc->Outputs().HasTag(kCompletionTag)) {
        ov::Tensor completion(ov::element::f32, {1});
        completion.data<float>()[0] = 1.0f;
        cc->Outputs().Tag(kCompletionTag).AddPacket(
            MakePacket<ov::Tensor>(std::move(completion))
                .At(Timestamp(frame_index_)));
      }
      return tool::StatusStop();
    }

    imp_tensor_t* infer_tensor = nullptr;
    imp_tensor_t* full_tensor = nullptr;
    if (emit_frame_ || emit_infer_) {
      infer_tensor = NextFrame(infer_stream_, &pending_infer_);
      if (!infer_tensor) return tool::StatusStop();
    }
    if (emit_full_) {
      full_tensor =
          NextFrame(shared_camera_stream_ ? infer_stream_ : full_stream_,
                    &pending_full_, shared_camera_stream_ ? 1 : 0);
      if (!full_tensor) {
        if (infer_tensor) imp_tensor_release(infer_tensor);
        return tool::StatusStop();
      }
    }

    const Timestamp timestamp(frame_index_);
    if (infer_tensor) {
      auto packet = MakeSurfacePacket(infer_tensor);
      RET_CHECK(packet) << "GstVideoSourceCalculator: surface frame failed";
      cc->Outputs()
          .Tag(emit_frame_ ? kFrameTag : kFrameInferTag)
          .Add(packet.release(), timestamp);
    }
    if (full_tensor) {
      auto packet = full_native_surface_ ? MakeSurfacePacket(full_tensor)
                                         : MakeHostNv12Packet(full_tensor);
      RET_CHECK(packet) << "GstVideoSourceCalculator: full frame failed";
      cc->Outputs().Tag(kFrameFullTag).Add(packet.release(), timestamp);
    }
    ++frame_index_;
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    ReleaseStream(&infer_ctx_, &infer_stream_, &pending_infer_);
    ReleaseStream(&full_ctx_, &full_stream_, &pending_full_);
    return absl::OkStatus();
  }

 private:
  // The MPI decode device is process-wide, not per stream.
  absl::Status ShareSessionDevice(CalculatorContext* cc) {
    if (!cc->InputSidePackets().HasTag(kSessionTag)) return absl::OkStatus();
    const OpenVINOSessionPtr& session =
        cc->InputSidePackets().Tag(kSessionTag).Get<OpenVINOSessionPtr>();
    RET_CHECK(session != nullptr) << "SESSION side packet is empty";
    if (session->remote_device_type == RemoteDeviceType::kNone) {
      return absl::OkStatus();
    }
    if (use_d3d11_) {
      RET_CHECK(session->remote_device_type == RemoteDeviceType::kD3D11)
          << "SESSION remote_context must be D3D11 on Windows";
      imp_video_set_d3d11_device(session->remote_device);
    } else {
      RET_CHECK(session->remote_device_type == RemoteDeviceType::kVa)
          << "SESSION remote_context must be VA on Linux";
      imp_video_set_va_display(session->remote_device);
    }
    return absl::OkStatus();
  }

  absl::Status OpenStream(imp_context_t** ctx, imp_video_stream_t** stream,
                          int w, int h, bool native_surface,
                          imp_tensor_t** pending,
                          imp_tensor_memory_type_t expected_first,
                          const imp_video_branch_t* branches = nullptr,
                          uint32_t branch_count = 0) {
    if (imp_context_create(ctx, nullptr) != IMP_OK) {
      return absl::InternalError("imp_context_create failed");
    }
    imp_video_source_t* src = nullptr;
    imp_video_source_create(&src, source_type_);
    if (source_type_ == IMP_SOURCE_CAMERA) {
      if (!camera_device_.empty()) {
        imp_video_source_set(src, "device", camera_device_.c_str());
      }
    } else {
      imp_video_source_set(src, source_type_ == IMP_SOURCE_FILE ? "path" : "url",
                           source_uri_.c_str());
    }
    if (w > 0) imp_video_source_set(src, "width", std::to_string(w).c_str());
    if (h > 0) imp_video_source_set(src, "height", std::to_string(h).c_str());

    imp_video_decode_opts_t options{};
    options.branches = branches;
    options.branch_count = branch_count;
    options.output_memory =
        native_surface ? IMP_VIDEO_MEMORY_NATIVE : IMP_VIDEO_MEMORY_SYSTEM;
    imp_status_t status = imp_video_open(stream, src, *ctx, &options);
    imp_video_source_destroy(src);
    if (status != IMP_OK) {
      return absl::InternalError(
          absl::StrCat("imp_video_open failed: ", imp_context_get_error(*ctx)));
    }
    status = imp_video_read_frame(pending, *stream, 0);
    if (status != IMP_OK || !*pending) {
      return absl::InternalError("first frame read failed");
    }
    if (imp_tensor_get_memory_type(*pending) != expected_first) {
      return absl::InternalError("unexpected frame memory type from GStreamer");
    }
    return absl::OkStatus();
  }

  imp_tensor_t* NextFrame(imp_video_stream_t* stream, imp_tensor_t** pending,
                          uint32_t branch = 0) {
    if (*pending) {
      imp_tensor_t* tensor = *pending;
      *pending = nullptr;
      return tensor;
    }
    imp_tensor_t* tensor = nullptr;
    if (imp_video_read_frame(&tensor, stream, branch) != IMP_OK) return nullptr;
    return tensor;
  }

  std::unique_ptr<GstVideoFramePacket> MakeSurfacePacket(
      imp_tensor_t* tensor) {
    auto packet = std::make_unique<GstVideoFramePacket>();
    int width = 0;
    int height = 0;
    if (use_d3d11_) {
      void* texture = nullptr;
      void* device = nullptr;
      uint32_t subresource = 0;
      if (imp_tensor_get_d3d11_texture(tensor, &texture, &device,
                                       &subresource, &width,
                                       &height) != IMP_OK) {
        imp_tensor_release(tensor);
        return nullptr;
      }
      packet->resource =
          D3D11Frame{reinterpret_cast<uintptr_t>(texture), subresource};
    } else {
      uint32_t surface_id = 0;
      void* display = nullptr;
      if (imp_tensor_get_va_surface(tensor, &surface_id, &display, &width,
                                    &height) != IMP_OK) {
        imp_tensor_release(tensor);
        return nullptr;
      }
      packet->resource = VaSurfaceFrame{surface_id, display};
    }
    packet->width = width;
    packet->height = height;
    packet->owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);
    return packet;
  }

  std::unique_ptr<GstVideoFramePacket> MakeHostNv12Packet(
      imp_tensor_t* tensor) {
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    int width = 0;
    int height = 0;
    if (imp_tensor_get_nv12_planes(tensor, &y, &uv, &width, &height) !=
        IMP_OK) {
      imp_tensor_release(tensor);
      return nullptr;
    }
    auto packet = std::make_unique<GstVideoFramePacket>();
    packet->resource = CpuNv12Frame{y, uv, width, width};
    packet->width = width;
    packet->height = height;
    packet->owner = std::shared_ptr<imp_tensor_t>(tensor, imp_tensor_release);
    return packet;
  }

  static void ReleaseStream(imp_context_t** ctx, imp_video_stream_t** stream,
                            imp_tensor_t** pending) {
    if (*pending) {
      imp_tensor_release(*pending);
      *pending = nullptr;
    }
    if (*stream) {
      imp_video_close(*stream);
      *stream = nullptr;
    }
    if (*ctx) {
      imp_context_destroy(*ctx);
      *ctx = nullptr;
    }
  }

  std::string source_uri_;
  std::string camera_device_;
  imp_source_type_t source_type_ = IMP_SOURCE_FILE;
  int width_ = 0;
  int height_ = 0;
  int full_width_ = 0;
  int full_height_ = 0;
  bool full_native_surface_ = false;
  int64_t max_frames_ = 0;
  int64_t frame_index_ = 0;
  bool use_d3d11_ = false;
  bool emit_frame_ = false;
  bool emit_infer_ = false;
  bool emit_full_ = false;
  bool shared_camera_stream_ = false;

  imp_context_t* infer_ctx_ = nullptr;
  imp_video_stream_t* infer_stream_ = nullptr;
  imp_tensor_t* pending_infer_ = nullptr;

  imp_context_t* full_ctx_ = nullptr;
  imp_video_stream_t* full_stream_ = nullptr;
  imp_tensor_t* pending_full_ = nullptr;
};

REGISTER_CALCULATOR(GstVideoSourceCalculator);

}  // namespace mediapipe
