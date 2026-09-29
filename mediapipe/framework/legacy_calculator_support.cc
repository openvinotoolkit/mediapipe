// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/legacy_calculator_support.h"

namespace mediapipe {

// Must match the ABSL_CONST_INIT (constinit) on the declarations; MSVC enforces it.
#ifndef __APPLE__
#define MP_LEGACY_CONST_INIT ABSL_CONST_INIT
#else
#define MP_LEGACY_CONST_INIT
#endif  // !__APPLE__

template <>
MP_LEGACY_CONST_INIT thread_local CalculatorContext*
    LegacyCalculatorSupport::Scoped<CalculatorContext>::current_ = nullptr;
template <>
MP_LEGACY_CONST_INIT thread_local CalculatorContract*
    LegacyCalculatorSupport::Scoped<CalculatorContract>::current_ = nullptr;

#undef MP_LEGACY_CONST_INIT

}  // namespace mediapipe
