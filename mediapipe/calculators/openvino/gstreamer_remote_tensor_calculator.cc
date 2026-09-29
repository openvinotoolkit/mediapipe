// Copyright 2026 The MediaPipe Authors.
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
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/openvino/gstreamer_remote_tensor_calculator.pb.h"
#include "mediapipe/calculators/openvino/openvino_session.h"
#include "mediapipe/calculators/openvino/remote_tensor_packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#include "openvino/openvino.hpp"

#if defined(MEDIAPIPE_ENABLE_GSTREAMER)
#include "gst/app/gstappsink.h"
#include "gst/gst.h"
#include "gst/va/gstvaallocator.h"
#include "gst/va/gstvadisplay.h"
#include "gst/va/gstvautils.h"
#include "gst/video/video.h"
#include "openvino/runtime/intel_gpu/ocl/va.hpp"
#define MEDIAPIPE_HAS_GSTREAMER 1
#else
#define MEDIAPIPE_HAS_GSTREAMER 0
#endif

namespace mediapipe {
namespace {

constexpr char kRemoteFrameTag[] = "REMOTE_FRAME";
constexpr char kSessionTag[] = "SESSION";

std::string DefaultGStreamerPipeline(const std::string& source,
                                     const std::string& caps) {
  std::string pipeline = "uridecodebin uri=\"" + source + "\"";
  const std::string output_caps = caps.empty()
      ? "video/x-raw(memory:VAMemory),format=NV12"
      : caps;
  pipeline += " ! capsfilter caps=\"" + output_caps + "\"";
  pipeline += " ! appsink name=remote_tensors_sink";
  return pipeline;
}

}  // namespace

// Emits NV12 VA surfaces using the same VA display as the OpenVINO session.
class GStreamerRemoteTensorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Tag(kRemoteFrameTag).Set<RemoteTensorPacket>();
    cc->InputSidePackets().Tag(kSessionTag).Set<
        openvino_calculators::OpenVINOSessionPtr>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<GStreamerRemoteTensorCalculatorOptions>();
    source_ = options.source();
    pipeline_ = options.pipeline();
    width_ = options.width();
    height_ = options.height();
    fps_ = options.fps();
    caps_ = options.caps();
    device_ = options.device();
    prefer_zero_copy_ = options.prefer_zero_copy();
    max_frames_ = options.max_frames();

#if MEDIAPIPE_HAS_GSTREAMER
    if (!gst_is_initialized()) {
      gst_init(nullptr, nullptr);
    }

    session_ = cc->InputSidePackets().Tag(kSessionTag).Get<
        openvino_calculators::OpenVINOSessionPtr>();
    if (session_ == nullptr || session_->native_va_display == nullptr) {
      return absl::FailedPreconditionError(
          "GStreamerRemoteTensorCalculator requires a SESSION compiled with "
          "va_device_path.");
    }
    try {
      va_context_ = std::make_unique<ov::intel_gpu::ocl::VAContext>(
          session_->compiled_model.get_context()
              .as<ov::intel_gpu::ocl::VAContext>());
    } catch (const std::exception& e) {
      return absl::FailedPreconditionError(absl::StrCat(
          "The OpenVINO session must use a VA-shared GPU context: ", e.what()));
    }

    if (pipeline_.empty() && source_.empty()) {
      return absl::FailedPreconditionError(
          "GStreamerRemoteTensorCalculator requires a non-empty `source` or "
          "`pipeline` option.");
    }

    if (pipeline_.empty()) {
      if (!gst_uri_is_valid(source_.c_str())) {
        const std::string absolute_path =
            std::filesystem::absolute(source_).string();
        GError* uri_error = nullptr;
        gchar* source_uri =
            gst_filename_to_uri(absolute_path.c_str(), &uri_error);
        if (source_uri == nullptr) {
          const std::string message = uri_error == nullptr
                                          ? "invalid source path"
                                          : uri_error->message;
          if (uri_error != nullptr) g_error_free(uri_error);
          return absl::InvalidArgumentError(
              absl::StrCat("Cannot convert source path to URI: ", message));
        }
        source_ = source_uri;
        g_free(source_uri);
      }
      pipeline_ = DefaultGStreamerPipeline(source_, caps_);
    }

    GError* error = nullptr;
    pipeline_element_ = gst_parse_launch(pipeline_.c_str(), &error);
    if (error != nullptr) {
      std::string message = error->message;
      g_error_free(error);
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse GStreamer pipeline '", pipeline_,
                       "': ", message));
    }
    if (pipeline_element_ == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to create GStreamer pipeline '", pipeline_,
                       "'"));
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_element_),
                     "remote_tensors_sink");
    if (appsink_ == nullptr) {
      return absl::FailedPreconditionError(
          "The GStreamer pipeline must expose an appsink element named "
          "'remote_tensors_sink' for remote-tensor extraction.");
    }

    GstVaDisplay* display = GST_VA_DISPLAY(session_->native_va_display);
    GstContext* gst_context = gst_context_new(
        GST_VA_DISPLAY_HANDLE_CONTEXT_TYPE_STR, TRUE);
    gst_context_set_va_display(gst_context, display);
    gst_element_set_context(pipeline_element_, gst_context);
    gst_context_unref(gst_context);

    g_object_set(appsink_, "sync", FALSE, "max-buffers", 2, "drop", TRUE,
                 nullptr);
    const GstStateChangeReturn state_result =
        gst_element_set_state(pipeline_element_, GST_STATE_PLAYING);
    if (state_result == GST_STATE_CHANGE_FAILURE) {
      return absl::InternalError(
          absl::StrCat("GStreamer pipeline failed to transition to PLAYING: ",
                       pipeline_));
    }

    return absl::OkStatus();
#else
    return absl::UnimplementedError(
        "GStreamer remote-tensor source is not available in this build; "
        "enable GStreamer headers and libraries to use the device-backed "
        "remote-tensor path.");
#endif
  }

  absl::Status Process(CalculatorContext* cc) override {
#if MEDIAPIPE_HAS_GSTREAMER
    if (max_frames_ > 0 && frame_count_ >= max_frames_) {
      return tool::StatusStop();
    }
    if (pipeline_element_ == nullptr || appsink_ == nullptr) {
      return absl::FailedPreconditionError(
          "GStreamer pipeline was not initialized for remote-tensor output.");
    }

    if (!prefer_zero_copy_) {
      return absl::UnimplementedError(
          "The host-copy GStreamer fallback is not implemented; enable "
          "prefer_zero_copy for VA-surface input.");
    }

    GstSample* sample = nullptr;
    while (sample == nullptr) {
      sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_),
                                            100 * GST_MSECOND);
      if (sample != nullptr) break;
      GstBus* bus = gst_element_get_bus(pipeline_element_);
      GstMessage* message = gst_bus_pop_filtered(
          bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
      gst_object_unref(bus);
      if (message != nullptr && GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &error, &debug);
        const gchar* source_name =
          GST_MESSAGE_SRC(message) == nullptr
            ? "unknown element"
            : GST_OBJECT_NAME(GST_MESSAGE_SRC(message));
        const std::string message_text = error == nullptr
                           ? "unknown error"
                           : absl::StrCat(
                               error->message, " (domain ",
                               error->domain, ", code ",
                               error->code, ")");
        const std::string debug_text =
          debug == nullptr ? "no additional debug details" : debug;
        if (error != nullptr) g_error_free(error);
        g_free(debug);
        gst_message_unref(message);
        return absl::InternalError(
          absl::StrCat("GStreamer pipeline failed in ", source_name, ": ",
                 message_text, "; debug: ", debug_text));
      }
      const bool reached_eos =
          message != nullptr && GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS;
      if (message != nullptr) gst_message_unref(message);
      if (reached_eos || gst_app_sink_is_eos(GST_APP_SINK(appsink_))) {
        return tool::StatusStop();
      }
    }
    std::shared_ptr<void> sample_owner(sample, [](void* value) {
      gst_sample_unref(static_cast<GstSample*>(value));
    });

    GstCaps* caps = gst_sample_get_caps(sample);
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstVideoInfo video_info;
    gst_video_info_init(&video_info);
    if (caps == nullptr || buffer == nullptr ||
        !gst_video_info_from_caps(&video_info, caps)) {
      return absl::InvalidArgumentError(
          "GStreamer appsink produced a sample without valid video caps");
    }
    const GstCapsFeatures* features = gst_caps_get_features(caps, 0);
    if (features == nullptr ||
        !gst_caps_features_contains(features, GST_CAPS_FEATURE_MEMORY_VA)) {
      return absl::FailedPreconditionError(
          "The appsink sample is not memory:VAMemory; refusing an implicit "
          "host copy. Constrain the pipeline caps to video/x-raw(memory:VAMemory).");
    }
    if (GST_VIDEO_INFO_FORMAT(&video_info) != GST_VIDEO_FORMAT_NV12) {
      return absl::InvalidArgumentError(
          "The VA remote-tensor path currently accepts NV12 samples only");
    }
    if ((width_ > 0 && width_ != GST_VIDEO_INFO_WIDTH(&video_info)) ||
        (height_ > 0 && height_ != GST_VIDEO_INFO_HEIGHT(&video_info))) {
      return absl::InvalidArgumentError(
          "The negotiated GStreamer dimensions do not match the configured "
          "width and height");
    }

    GstVaDisplay* display = gst_va_buffer_peek_display(buffer);
    if (display == nullptr || gst_va_display_get_va_dpy(display) !=
                    static_cast<VADisplay>(*va_context_)) {
      return absl::FailedPreconditionError(
          "The decoded VA surface does not use the OpenVINO session's VA "
          "display");
    }
    const VASurfaceID surface = gst_va_buffer_get_surface(buffer);
    const size_t width = GST_VIDEO_INFO_WIDTH(&video_info);
    const size_t height = GST_VIDEO_INFO_HEIGHT(&video_info);
    try {
      auto planes = va_context_->create_tensor_nv12(height, width, surface);
      auto packet = std::make_unique<RemoteTensorPacket>();
      packet->tensors.push_back(planes.first);
      packet->tensors.push_back(planes.second);
      packet->nv12_two_planes = true;
      packet->native_owner = std::move(sample_owner);
      const GstClockTime pts = GST_BUFFER_PTS(buffer);
      const int64_t pts_us = GST_CLOCK_TIME_IS_VALID(pts)
                                 ? static_cast<int64_t>(pts / 1000)
                                 : last_timestamp_us_ + 1;
      last_timestamp_us_ = std::max(last_timestamp_us_ + 1, pts_us);
      const Timestamp timestamp =
          Timestamp::FromMicroseconds(last_timestamp_us_);
      cc->Outputs().Tag(kRemoteFrameTag).Add(packet.release(), timestamp);
      ++frame_count_;
      return absl::OkStatus();
    } catch (const std::exception& e) {
      return absl::InternalError(absl::StrCat(
          "Failed to import the NV12 VA surface into OpenVINO: ", e.what()));
    }
#else
    return absl::UnimplementedError(
        "GStreamer remote-tensor source requires a GStreamer-enabled build and "
      "OpenVINO remote-context support.");
#endif
  }

  absl::Status Close(CalculatorContext* cc) override {
#if MEDIAPIPE_HAS_GSTREAMER
    if (appsink_ != nullptr) {
      gst_object_unref(appsink_);
      appsink_ = nullptr;
    }
    if (pipeline_element_ != nullptr) {
      gst_element_set_state(pipeline_element_, GST_STATE_NULL);
      gst_object_unref(pipeline_element_);
      pipeline_element_ = nullptr;
    }
#endif
    return absl::OkStatus();
  }

 private:
  std::string source_;
  std::string pipeline_;
  std::string caps_;
  std::string device_;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 30.0;
  bool prefer_zero_copy_ = true;
  int max_frames_ = 0;
  int frame_count_ = 0;
  int64_t last_timestamp_us_ = -1;
  openvino_calculators::OpenVINOSessionPtr session_;
#if MEDIAPIPE_HAS_GSTREAMER
  std::unique_ptr<ov::intel_gpu::ocl::VAContext> va_context_;
  GstElement* pipeline_element_ = nullptr;
  GstElement* appsink_ = nullptr;
#endif
};

REGISTER_CALCULATOR(GStreamerRemoteTensorCalculator);

}  // namespace mediapipe
