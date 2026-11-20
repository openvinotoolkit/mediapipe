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
#include <cstdint>

#include "mediapipe/calculators/gstreamer/gst_detection_box.h"
#include "mediapipe/calculators/gstreamer/gst_nv12_overlay_draw.h"
#include "mediapipe/calculators/gstreamer/gst_video_frame_packet.h"
#include "mediapipe/calculators/gstreamer/mpi/intel_mpi.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"

#ifndef _WIN32
#include <va/va.h>
#endif

namespace mediapipe {
namespace {

constexpr char kFrameFullTag[] = "FRAME_FULL";
constexpr char kDetectionBoxesTag[] = "DETECTION_BOXES";
constexpr char kFrameOverlaidTag[] = "FRAME_OVERLAID";
constexpr char kColorRTag[] = "COLOR_R";
constexpr char kColorGTag[] = "COLOR_G";
constexpr char kColorBTag[] = "COLOR_B";

int ReadColorComponent(const Packet& packet) {
  const int64_t value = packet.ValidateAsType<int64_t>().ok()
                            ? packet.Get<int64_t>()
                            : packet.Get<int>();
  return static_cast<int>(std::clamp<int64_t>(value, 0, 255));
}

nv12overlay::Color RgbToNv12(int red, int green, int blue) {
  const float r = static_cast<float>(red);
  const float g = static_cast<float>(green);
  const float b = static_cast<float>(blue);
  return {static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b),
          static_cast<uint8_t>(-0.168736f * r - 0.331264f * g + 0.5f * b +
                               128.0f),
          static_cast<uint8_t>(0.5f * r - 0.418688f * g - 0.081312f * b +
                               128.0f)};
}

}  // namespace

// Draws detection boxes in place on a full-resolution NV12 frame and forwards
// the same packet. Host NV12 frames are written directly; VA surfaces (Linux)
// are mapped with vaDeriveImage, or copied through vaGetImage/vaPutImage when
// the driver cannot derive the surface. FRAME_FULL must have no other
// consumer, since the frame is modified in place.
//
// Inputs:
//   FRAME_FULL      - GstVideoFramePacket.
//   DETECTION_BOXES - GstDetectionBoxes in FRAME_FULL pixels.
// Output:
//   FRAME_OVERLAID  - GstVideoFramePacket, the annotated FRAME_FULL.
// Input side packets (optional, all or none):
//   COLOR_R, COLOR_G, COLOR_B - int or int64 box color, green by default.
//
//   node {
//     calculator: "GstDetectionOverlayCalculator"
//     input_stream: "FRAME_FULL:frame_full"
//     input_stream: "DETECTION_BOXES:boxes"
//     output_stream: "FRAME_OVERLAID:frame_overlaid"
//   }
class GstDetectionOverlayCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kFrameFullTag).Set<GstVideoFramePacket>();
    cc->Inputs().Tag(kDetectionBoxesTag).Set<GstDetectionBoxes>();
    cc->Outputs().Tag(kFrameOverlaidTag).Set<GstVideoFramePacket>();
    for (const char* tag : {kColorRTag, kColorGTag, kColorBTag}) {
      if (cc->InputSidePackets().HasTag(tag)) {
        cc->InputSidePackets().Tag(tag).SetOneOf<int, int64_t>();
      }
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& side = cc->InputSidePackets();
    const bool has_red = side.HasTag(kColorRTag);
    RET_CHECK(has_red == side.HasTag(kColorGTag) &&
              has_red == side.HasTag(kColorBTag))
        << "COLOR_R, COLOR_G, and COLOR_B must be provided together";
    if (has_red) {
      color_ = RgbToNv12(ReadColorComponent(side.Tag(kColorRTag)),
                         ReadColorComponent(side.Tag(kColorGTag)),
                         ReadColorComponent(side.Tag(kColorBTag)));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& frame =
        cc->Inputs().Tag(kFrameFullTag).Get<GstVideoFramePacket>();
    const auto& boxes =
        cc->Inputs().Tag(kDetectionBoxesTag).Get<GstDetectionBoxes>();
    imp_tensor_t* tensor = frame.owner.get();
    RET_CHECK(tensor != nullptr) << "FRAME_FULL has no native tensor owner";

    const imp_tensor_memory_type_t memory = imp_tensor_get_memory_type(tensor);
    if (memory == IMP_MEM_SYSTEM) {
      ABSL_RETURN_IF_ERROR(DrawOnHost(tensor, boxes));
    } else if (memory == IMP_MEM_VA_SURFACE) {
#ifdef _WIN32
      RET_CHECK_FAIL() << "VA surfaces are not supported on Windows";
#else
      if (!boxes.empty()) ABSL_RETURN_IF_ERROR(DrawOnVaSurface(tensor, boxes));
#endif
    } else {
      RET_CHECK_FAIL() << "overlay supports host NV12 or VA-surface frames";
    }

    cc->Outputs().Tag(kFrameOverlaidTag).AddPacket(
        cc->Inputs().Tag(kFrameFullTag).Value());
    return absl::OkStatus();
  }

 private:
  absl::Status DrawOnHost(imp_tensor_t* tensor,
                          const GstDetectionBoxes& boxes) {
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    int w = 0;
    int h = 0;
    RET_CHECK_EQ(imp_tensor_get_nv12_planes(tensor, &y, &uv, &w, &h), IMP_OK);
    RET_CHECK(w > 0 && h > 0);
    // FRAME_FULL has a single consumer, so writing the planes is safe.
    nv12overlay::DrawDetections(const_cast<uint8_t*>(y), w,
                                const_cast<uint8_t*>(uv), w, w, h, boxes,
                                color_);
    return absl::OkStatus();
  }

#ifndef _WIN32
  absl::Status DrawOnVaSurface(imp_tensor_t* tensor,
                               const GstDetectionBoxes& boxes) {
    uint32_t surface_id = 0;
    void* display = nullptr;
    int w = 0;
    int h = 0;
    RET_CHECK_EQ(imp_tensor_get_va_surface(tensor, &surface_id, &display, &w,
                                           &h),
                 IMP_OK);
    RET_CHECK(display != nullptr) << "VA surface has no VADisplay";
    RET_CHECK(w > 0 && h > 0);

    VADisplay va = static_cast<VADisplay>(display);
    const VASurfaceID surface = static_cast<VASurfaceID>(surface_id);
    vaSyncSurface(va, surface);

    VAImage image{};
    const bool derived = vaDeriveImage(va, surface, &image) == VA_STATUS_SUCCESS;
    if (!derived) {
      VAImageFormat format{};
      format.fourcc = VA_FOURCC_NV12;
      format.byte_order = VA_LSB_FIRST;
      format.bits_per_pixel = 12;
      RET_CHECK_EQ(vaCreateImage(va, &format, w, h, &image), VA_STATUS_SUCCESS)
          << "vaCreateImage(NV12) failed";
      if (vaGetImage(va, surface, 0, 0, w, h, image.image_id) !=
          VA_STATUS_SUCCESS) {
        vaDestroyImage(va, image.image_id);
        return absl::InternalError("vaGetImage failed");
      }
    }

    void* buffer = nullptr;
    if (vaMapBuffer(va, image.buf, &buffer) != VA_STATUS_SUCCESS) {
      vaDestroyImage(va, image.image_id);
      return absl::InternalError("vaMapBuffer failed");
    }
    auto* base = static_cast<uint8_t*>(buffer);
    nv12overlay::DrawDetections(base + image.offsets[0], image.pitches[0],
                                base + image.offsets[1], image.pitches[1], w,
                                h, boxes, color_);
    vaUnmapBuffer(va, image.buf);
    if (!derived) {
      vaPutImage(va, surface, image.image_id, 0, 0, w, h, 0, 0, w, h);
    }
    vaDestroyImage(va, image.image_id);
    return absl::OkStatus();
  }
#endif

  nv12overlay::Color color_ = nv12overlay::kGreen;
};

REGISTER_CALCULATOR(GstDetectionOverlayCalculator);

}  // namespace mediapipe
