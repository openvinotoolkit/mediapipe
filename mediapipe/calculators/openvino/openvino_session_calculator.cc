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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "mediapipe/calculators/openvino/openvino_session.h"
#include "mediapipe/calculators/openvino/openvino_session_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "nlohmann/json.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
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
  return options.SerializeAsString();
}

ov::AnyMap ToAnyMap(const OpenVINOSessionCalculatorOptions& options) {
  ov::AnyMap config;
  for (const auto& entry : options.plugin_config()) {
    config[entry.key()] = entry.value();
  }
  return config;
}

std::string Trim(absl::string_view value) {
  return std::string(absl::StripAsciiWhitespace(value));
}

std::vector<std::string> ParsePair(const std::string& value,
                                   const char* option_name) {
  std::vector<std::string> parts = absl::StrSplit(value, ':');
  if (parts.empty() || parts.size() > 2) {
    throw std::invalid_argument(
        absl::StrCat(option_name, " must contain one value or one ':' pair"));
  }
  for (std::string& part : parts) {
    part = Trim(part);
    if (part.empty()) {
      throw std::invalid_argument(
          absl::StrCat(option_name, " contains an empty value"));
    }
  }
  return parts;
}

ov::PartialShape ParseShape(const nlohmann::json& value,
                            const ov::Rank& input_rank) {
  if (value.is_string()) {
    std::string text = Trim(value.get<std::string>());
    std::string lower = absl::AsciiStrToLower(text);
    if (lower == "auto") {
      return ov::PartialShape::dynamic(input_rank);
    }
    if (text.size() < 2 ||
        !((text.front() == '(' && text.back() == ')') ||
          (text.front() == '[' && text.back() == ']'))) {
      throw std::invalid_argument(
          absl::StrCat("invalid shape value: ", text));
    }
    text = text.substr(1, text.size() - 2);
    std::vector<ov::Dimension> dimensions;
    for (absl::string_view token : absl::StrSplit(text, ',')) {
      const std::string dimension = Trim(token);
      int64_t size;
      if (dimension == "?" || dimension == "-1" ||
          absl::AsciiStrToLower(dimension) == "auto") {
        dimensions.emplace_back(ov::Dimension::dynamic());
      } else if (!absl::SimpleAtoi(dimension, &size) || size < 0) {
        throw std::invalid_argument(
            absl::StrCat("invalid shape dimension: ", dimension));
      } else {
        dimensions.emplace_back(size);
      }
    }
    return ov::PartialShape(dimensions);
  }
  if (value.is_array()) {
    std::vector<ov::Dimension> dimensions;
    for (const auto& dimension : value) {
      if (!dimension.is_number_integer() || dimension.get<int64_t>() < 0) {
        throw std::invalid_argument(
            "shape array dimensions must be nonnegative integers");
      }
      dimensions.emplace_back(dimension.get<int64_t>());
    }
    return ov::PartialShape(dimensions);
  }
  throw std::invalid_argument("shape must be a string or array");
}

bool ApplyShape(const OpenVINOSessionCalculatorOptions& options,
                std::shared_ptr<ov::Model>& model) {
  if (!options.shape().empty()) {
    const std::string shape = Trim(options.shape());
    std::map<std::string, ov::PartialShape> shapes;
    if (!shape.empty() && shape.front() == '{') {
      const auto parsed = nlohmann::json::parse(shape);
      if (!parsed.is_object()) {
        throw std::invalid_argument("shape JSON must be an object");
      }
      for (const auto& [name, value] : parsed.items()) {
        const auto input = model->input(name);
        shapes[name] = ParseShape(value, input.get_partial_shape().rank());
      }
    } else {
      if (model->inputs().size() != 1) {
        throw std::invalid_argument(
            "an unnamed shape can only be used with a single-input model");
      }
      const auto input = model->input();
      shapes[input.get_any_name()] =
          ParseShape(shape, input.get_partial_shape().rank());
    }
    model->reshape(shapes);
    return true;
  }
  return false;
}

void ApplyBatch(const OpenVINOSessionCalculatorOptions& options,
                std::shared_ptr<ov::Model>& model) {
  if (!options.batch_size().empty()) {
    for (auto& input : model->inputs()) {
      const ov::Layout layout = ov::layout::get_layout(input);
      if (!ov::layout::has_batch(layout)) {
        ov::layout::set_layout(input, ov::Layout("[N,...]"));
      }
    }
    const std::string batch_size = Trim(options.batch_size());
    if (absl::AsciiStrToLower(batch_size) == "auto") {
      ov::set_batch(model, ov::Dimension::dynamic());
      return;
    }
    int64_t size;
    if (!absl::SimpleAtoi(batch_size, &size) || size <= 0) {
      throw std::invalid_argument(
          "batch_size must be a positive integer or auto");
    }
    ov::set_batch(model, size);
  }
}

bool HasInput(const std::shared_ptr<ov::Model>& model,
              const std::string& name) {
  for (const auto& input : model->inputs()) {
    if (input.get_names().count(name) > 0) return true;
  }
  return false;
}

void ApplyLayout(ov::preprocess::PrePostProcessor& preprocessor,
                 const std::shared_ptr<ov::Model>& model,
                 const std::string& configured_layout) {
  if (configured_layout.empty()) return;

  auto apply = [](auto& info, const std::string& value) {
    const std::vector<std::string> layouts = ParsePair(value, "layout");
    info.tensor().set_layout(ov::Layout(layouts[0]));
    info.model().set_layout(
        ov::Layout(layouts.size() == 2 ? layouts[1] : layouts[0]));
  };

  const std::string layout = Trim(configured_layout);
  if (!layout.empty() && layout.front() == '{') {
    const auto parsed = nlohmann::json::parse(layout);
    if (!parsed.is_object()) {
      throw std::invalid_argument("layout JSON must be an object");
    }
    for (const auto& [name, value] : parsed.items()) {
      if (!value.is_string()) {
        throw std::invalid_argument("layout values must be strings");
      }
      if (HasInput(model, name)) {
        apply(preprocessor.input(name), value.get<std::string>());
      } else {
        model->output(name);
        apply(preprocessor.output(name), value.get<std::string>());
      }
    }
  } else {
    if (model->inputs().size() != 1) {
      throw std::invalid_argument(
          "an unnamed layout can only be used with a single-input model");
    }
    apply(preprocessor.input(), layout);
  }
}

std::vector<float> ParseFloatValues(const std::string& configured,
                                    const char* option_name) {
  std::string value = Trim(configured);
  if (value.empty()) return {};
  if ((value.front() == '(' && value.back() == ')') ||
      (value.front() == '[' && value.back() == ']')) {
    value = value.substr(1, value.size() - 2);
  }
  std::vector<float> values;
  for (absl::string_view token : absl::StrSplit(value, ',')) {
    float parsed;
    if (!absl::SimpleAtof(Trim(token), &parsed)) {
      throw std::invalid_argument(
          absl::StrCat("invalid ", option_name, " value: ", token));
    }
    values.push_back(parsed);
  }
  return values;
}

ov::preprocess::ColorFormat ParseColorFormat(const std::string& value) {
  const std::string format = absl::AsciiStrToUpper(Trim(value));
  if (format == "RGB") return ov::preprocess::ColorFormat::RGB;
  if (format == "BGR") return ov::preprocess::ColorFormat::BGR;
  if (format == "GRAY") return ov::preprocess::ColorFormat::GRAY;
  if (format == "NV12") return ov::preprocess::ColorFormat::NV12_SINGLE_PLANE;
  if (format == "NV12_2") return ov::preprocess::ColorFormat::NV12_TWO_PLANES;
  if (format == "I420") return ov::preprocess::ColorFormat::I420_SINGLE_PLANE;
  if (format == "I420_3") return ov::preprocess::ColorFormat::I420_THREE_PLANES;
  throw std::invalid_argument(absl::StrCat("unsupported color_format: ", value));
}

ov::element::Type ParsePrecision(const std::string& value) {
  std::string precision = absl::AsciiStrToLower(Trim(value));
  if (precision == "fp64") precision = "f64";
  if (precision == "fp32") precision = "f32";
  if (precision == "fp16") precision = "f16";
  if (precision.rfind("uint", 0) == 0) precision.replace(0, 4, "u");
  if (precision.rfind("int", 0) == 0) precision.replace(0, 3, "i");
  const ov::element::Type type(precision);
  if (!type.is_static()) {
    throw std::invalid_argument(absl::StrCat("unsupported precision: ", value));
  }
  return type;
}

std::shared_ptr<ov::Model> PrepareModel(
    const OpenVINOSessionCalculatorOptions& options) {
  std::shared_ptr<ov::Model> model =
      GlobalCore().read_model(options.model_path());
  const bool shape_applied = ApplyShape(options, model);

  ov::preprocess::PrePostProcessor preprocessor(model);
  ApplyLayout(preprocessor, model, options.layout());
  if (!options.mean().empty() || !options.scale().empty() ||
      !options.color_format().empty() || !options.precision().empty()) {
    if (model->inputs().size() != 1) {
      throw std::invalid_argument(
          "mean, scale, color_format, and precision require a single-input "
          "model");
    }
    auto& input = preprocessor.input();
    if (!options.color_format().empty()) {
      const auto formats = ParsePair(options.color_format(), "color_format");
      input.tensor().set_color_format(ParseColorFormat(formats[0]));
      if (formats.size() == 2) {
        input.preprocess().convert_color(ParseColorFormat(formats[1]));
      }
    }
    if (!options.precision().empty()) {
      const auto precisions = ParsePair(options.precision(), "precision");
      input.tensor().set_element_type(ParsePrecision(precisions[0]));
      input.preprocess().convert_element_type(
          precisions.size() == 2 ? ParsePrecision(precisions[1])
                                 : model->input().get_element_type());
    }
    const std::vector<float> means = ParseFloatValues(options.mean(), "mean");
    if (means.size() == 1) {
      input.preprocess().mean(means[0]);
    } else if (!means.empty()) {
      input.preprocess().mean(means);
    }
    const std::vector<float> scales = ParseFloatValues(options.scale(), "scale");
    if (scales.size() == 1) {
      input.preprocess().scale(scales[0]);
    } else if (!scales.empty()) {
      input.preprocess().scale(scales);
    }
  }
  model = preprocessor.build();
  if (!shape_applied) ApplyBatch(options, model);
  return model;
}

}  // namespace

// Compiles a model with OpenVINO and publishes it, together with a queue of
// inference requests, as an output side packet consumed by
// OpenVINOInferenceCalculator.
//
// Node parameters: model_path, device, plugin_config, num_infer_requests,
// shape, batch_size, layout, mean, scale, color_format, precision.
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
      std::shared_ptr<ov::Model> model = PrepareModel(options);
      ov::CompiledModel compiled = GlobalCore().compile_model(
          model, options.device(), ToAnyMap(options));
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
