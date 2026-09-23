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

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/openvino/openvino_session.h"
#include "mediapipe/calculators/openvino/openvino_session_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "openvino/openvino.hpp"

namespace mediapipe {
namespace {

using ::mediapipe::openvino_calculators::OpenVINOSession;
using ::mediapipe::openvino_calculators::OpenVINOSessionPtr;

constexpr char kSessionTag[] = "SESSION";

// Process-wide ov::Core. Creating more than one is wasteful and defeats the
// model caching that the plugins do internally.
ov::Core& GlobalCore() {
  static ov::Core* core = new ov::Core();
  return *core;
}

std::mutex& SessionRegistryMutex() {
  static std::mutex* mutex = new std::mutex();
  return *mutex;
}

// Sessions are keyed by the full set of options that influences compilation,
// so that separate graphs asking for the same model share one compiled model
// and one inference request queue.
std::map<std::string, std::weak_ptr<OpenVINOSession>>& SessionRegistry() {
  static auto* registry =
      new std::map<std::string, std::weak_ptr<OpenVINOSession>>();
  return *registry;
}

std::string BuildSessionKey(const OpenVINOSessionCalculatorOptions& options) {
  std::string key = absl::StrCat(options.model_path(), "|", options.device(),
                                 "|", options.num_infer_requests(), "|");
  for (const auto& entry : options.plugin_config()) {
    absl::StrAppend(&key, entry.key(), "=", entry.value(), ";");
  }
  return key;
}

ov::AnyMap ToAnyMap(const OpenVINOSessionCalculatorOptions& options) {
  ov::AnyMap config;
  for (const auto& entry : options.plugin_config()) {
    config[entry.key()] = entry.value();
  }
  return config;
}

}  // namespace

// Compiles a model with OpenVINO and publishes it, together with a queue of
// inference requests, as an output side packet consumed by
// OpenVINOInferenceCalculator.
//
// Node parameters: model_path, device, plugin_config, num_infer_requests.
//
//   node {
//     calculator: "OpenVINOSessionCalculator"
//     output_side_packet: "SESSION:session"
//     node_options: {
//       [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
//         model_path: "/models/resnet50.xml"
//         device: "CPU"
//         plugin_config { key: "PERFORMANCE_HINT" value: "THROUGHPUT" }
//       }
//     }
//   }
class OpenVINOSessionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().GetTags().empty());
    RET_CHECK(cc->Outputs().GetTags().empty());
    cc->OutputSidePackets().Tag(kSessionTag).Set<OpenVINOSessionPtr>();
    const auto& options = cc->Options<OpenVINOSessionCalculatorOptions>();
    RET_CHECK(!options.model_path().empty())
        << "OpenVINOSessionCalculator requires model_path";
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<OpenVINOSessionCalculatorOptions>();
    const std::string key = BuildSessionKey(options);

    std::lock_guard<std::mutex> lock(SessionRegistryMutex());
    auto& registry = SessionRegistry();
    auto it = registry.find(key);
    if (it != registry.end()) {
      if (OpenVINOSessionPtr existing = it->second.lock()) {
        cc->OutputSidePackets()
            .Tag(kSessionTag)
            .Set(MakePacket<OpenVINOSessionPtr>(std::move(existing)));
        return absl::OkStatus();
      }
      registry.erase(it);
    }

    OpenVINOSessionPtr session;
    try {
      ov::CompiledModel compiled = GlobalCore().compile_model(
          options.model_path(), options.device(), ToAnyMap(options));
      int nireq = static_cast<int>(options.num_infer_requests());
      if (nireq <= 0) {
        nireq = static_cast<int>(
            compiled.get_property(ov::optimal_number_of_infer_requests));
      }
      if (nireq <= 0) {
        nireq = 1;
      }
      session = std::make_shared<OpenVINOSession>(std::move(compiled), nireq);
      ABSL_LOG(INFO) << "OpenVINOSessionCalculator compiled "
                     << options.model_path() << " on " << options.device()
                     << " with " << nireq << " inference requests";
    } catch (const std::exception& e) {
      return absl::InternalError(
          absl::StrCat("OpenVINOSessionCalculator failed to compile model '",
                       options.model_path(), "' on device '", options.device(),
                       "': ", e.what()));
    }

    registry[key] = session;
    cc->OutputSidePackets()
        .Tag(kSessionTag)
        .Set(MakePacket<OpenVINOSessionPtr>(std::move(session)));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(OpenVINOSessionCalculator);

}  // namespace mediapipe
