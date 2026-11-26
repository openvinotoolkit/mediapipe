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

#ifndef MEDIAPIPE_CALCULATORS_GSTREAMER_GST_VIDEO_FRAME_PACKET_H_
#define MEDIAPIPE_CALCULATORS_GSTREAMER_GST_VIDEO_FRAME_PACKET_H_

#include <cstdint>
#include <memory>
#include <variant>

// Defined in mpi/intel_mpi.h. The source constructs the owner with
// imp_tensor_release as its deleter, so consumers that only retain ownership
// (e.g. OpenVINOInferenceCalculator) do not depend on GStreamer.
extern "C" {
typedef struct imp_tensor_s imp_tensor_t;
}

namespace mediapipe {

struct VaSurfaceFrame {
  uint32_t surface_id = 0;
  void* display = nullptr;
};

struct D3D11Frame {
  uintptr_t texture_handle = 0;
  uint32_t subresource = 0;
};

// Packed host NV12 frame (stride == width).
struct CpuNv12Frame {
  const uint8_t* y = nullptr;
  const uint8_t* uv = nullptr;
  int y_stride = 0;
  int uv_stride = 0;
};

using GstVideoFrameResource =
    std::variant<VaSurfaceFrame, D3D11Frame, CpuNv12Frame>;

// One decoded NV12 frame. `owner` keeps the native frame alive and is the
// canonical data accessor for overlay/sink; `resource` is a non-owning handle
// snapshot for inference and must not outlive `owner`.
struct GstVideoFramePacket {
  GstVideoFrameResource resource = VaSurfaceFrame{};
  std::shared_ptr<imp_tensor_t> owner;
  int width = 0;
  int height = 0;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_GSTREAMER_GST_VIDEO_FRAME_PACKET_H_
