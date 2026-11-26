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

#ifndef MEDIAPIPE_CALCULATORS_GSTREAMER_GST_NV12_OVERLAY_DRAW_H_
#define MEDIAPIPE_CALCULATORS_GSTREAMER_GST_NV12_OVERLAY_DRAW_H_

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "mediapipe/calculators/gstreamer/gst_detection_box.h"

namespace mediapipe {
namespace nv12overlay {

struct Color {
  uint8_t y;
  uint8_t u;
  uint8_t v;
};

constexpr Color kGreen{150, 44, 21};
constexpr uint8_t kWhiteY = 235, kWhiteU = 128, kWhiteV = 128;
constexpr int kBorderThickness = 3;

// Fills an inclusive rectangle. Strides are per-plane row pitches, so the same
// code serves packed host NV12 and mapped VA images.
inline void FillRect(uint8_t* y, int y_stride, uint8_t* uv, int uv_stride,
                     int w, int h, int x1, int y1, int x2, int y2, uint8_t fy,
                     uint8_t fu, uint8_t fv) {
  x1 = std::clamp(x1, 0, w - 1);
  y1 = std::clamp(y1, 0, h - 1);
  x2 = std::clamp(x2, 0, w - 1);
  y2 = std::clamp(y2, 0, h - 1);
  if (x2 < x1 || y2 < y1) return;
  for (int row = y1; row <= y2; ++row) {
    std::memset(&y[row * y_stride + x1], fy,
                static_cast<size_t>(x2 - x1 + 1));
  }
  for (int row = y1 / 2; row <= y2 / 2; ++row) {
    for (int col = x1 / 2; col <= x2 / 2; ++col) {
      uv[row * uv_stride + col * 2] = fu;
      uv[row * uv_stride + col * 2 + 1] = fv;
    }
  }
}

// Draws box borders for every detection plus a confidence bar for the first
// box.
inline void DrawDetections(uint8_t* y, int y_stride, uint8_t* uv,
                           int uv_stride, int w, int h,
                           const GstDetectionBoxes& boxes,
                           const Color& color = kGreen) {
  if (boxes.empty()) return;
  for (const auto& d : boxes) {
    const int x1 = std::clamp(static_cast<int>(d.xmin), 0, w - 1);
    const int y1 = std::clamp(static_cast<int>(d.ymin), 0, h - 1);
    const int x2 = std::clamp(static_cast<int>(d.xmax), 0, w - 1);
    const int y2 = std::clamp(static_cast<int>(d.ymax), 0, h - 1);
    const int t = kBorderThickness;
    FillRect(y, y_stride, uv, uv_stride, w, h, x1, y1, x2,
             std::min(y1 + t - 1, h - 1), color.y, color.u, color.v);
    FillRect(y, y_stride, uv, uv_stride, w, h, x1, std::max(y2 - t + 1, 0), x2,
             y2, color.y, color.u, color.v);
    FillRect(y, y_stride, uv, uv_stride, w, h, x1, y1,
             std::min(x1 + t - 1, w - 1), y2, color.y, color.u, color.v);
    FillRect(y, y_stride, uv, uv_stride, w, h, std::max(x2 - t + 1, 0), y1, x2,
             y2, color.y, color.u, color.v);
  }

  const float best = boxes.front().score;
  if (best > 0.1f) {
    const int bar_w = static_cast<int>(best * 200.0f);
    FillRect(y, y_stride, uv, uv_stride, w, h, 4, 4, 208, 16, 16, 128, 128);
    FillRect(y, y_stride, uv, uv_stride, w, h, 4, 4, 4 + bar_w, 16, color.y,
             color.u, color.v);
    FillRect(y, y_stride, uv, uv_stride, w, h, 3, 3, 209, 3, kWhiteY, kWhiteU,
             kWhiteV);
    FillRect(y, y_stride, uv, uv_stride, w, h, 3, 17, 209, 17, kWhiteY,
             kWhiteU, kWhiteV);
    FillRect(y, y_stride, uv, uv_stride, w, h, 3, 3, 3, 17, kWhiteY, kWhiteU,
             kWhiteV);
    FillRect(y, y_stride, uv, uv_stride, w, h, 209, 3, 209, 17, kWhiteY,
             kWhiteU, kWhiteV);
  }
}

}  // namespace nv12overlay
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_GSTREAMER_GST_NV12_OVERLAY_DRAW_H_
