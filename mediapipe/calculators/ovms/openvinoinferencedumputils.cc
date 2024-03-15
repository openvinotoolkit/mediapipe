//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <ctime>
#include <chrono>
#include <sstream>
#include <string>
#include <fstream>
#include <openvino/openvino.hpp>

#include "mediapipe/calculators/ovms/openvinoinferencedumputils.h"

namespace mediapipe {
    
int INPUT_COUNTER = 1;

using InferenceInput = std::map<std::string, ov::Tensor>;

static std::stringstream dumpOvTensor(const ov::Tensor& tensor) {
    std::stringstream dumpStream;
    switch (tensor.get_element_type()) {
        // The cosine window and square root of Hann are equivalent.
        case ov::element::Type_t::f64: {
            const _Float64* input_tensor_access = reinterpret_cast<_Float64*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::f32: {
            const _Float32* input_tensor_access = reinterpret_cast<_Float32*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::i64: {
            const int64_t* input_tensor_access = reinterpret_cast<int64_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::i32: {
            const int32_t* input_tensor_access = reinterpret_cast<int32_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::i16: {
            const int16_t* input_tensor_access = reinterpret_cast<int16_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::i8: {
            const int8_t* input_tensor_access = reinterpret_cast<int8_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::u64: {
            const u_int64_t* input_tensor_access = reinterpret_cast<uint64_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::u32: {
            const u_int32_t* input_tensor_access = reinterpret_cast<uint32_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::u16: {
            const u_int16_t* input_tensor_access = reinterpret_cast<uint16_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::u8: {
            const u_int8_t* input_tensor_access = reinterpret_cast<uint8_t*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::boolean: {
            const bool* input_tensor_access = reinterpret_cast<bool*>(tensor.data());
            dumpStream << " tensor: [ ";
            for (int x = 0; x < tensor.get_size(); x++) {
                dumpStream << &input_tensor_access[x] << " ";
            }
            dumpStream << " ]";
            break;
        }
        case ov::element::Type_t::bf16:
        case ov::element::Type_t::undefined:
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::f16:
        case ov::element::Type_t::i4:
        case ov::element::Type_t::u4:
        case ov::element::Type_t::u1: {
            dumpStream << " unsupported dump type: [ " << tensor.get_element_type() << " ]";
            break;
        }
    }

    return dumpStream;    
}

static bool isAbsolutePath(const std::string& path) {
        return !path.empty() && (path[0] == '/');
}

static std::string joinPath(std::initializer_list<std::string> segments) {
    std::string joined;

    for (const auto& seg : segments) {
        if (joined.empty()) {
            joined = seg;
        } else if (isAbsolutePath(seg)) {
            if (joined[joined.size() - 1] == '/') {
                joined.append(seg.substr(1));
            } else {
                joined.append(seg);
            }
        } else {
            if (joined[joined.size() - 1] != '/') {
                joined.append("/");
            }
            joined.append(seg);
        }
    }

    return joined;
}

static void writeToFile(std::stringstream& stream, std::string name)
{
    std::ofstream ofs;
    ofs.open(name);
    ofs << stream.rdbuf();
    ofs.close();

    return;
}

static std::string getTimestampString() {
    time_t *rawtime = new time_t;
    struct tm * timeinfo;
    time(rawtime);
    timeinfo = localtime(rawtime);
    auto start = std::chrono::system_clock::now();
    std::stringstream timestampStream;
    timestampStream << timeinfo->tm_year << "_" << timeinfo->tm_mon << "_" << timeinfo->tm_mday << "_" ;
    timestampStream << timeinfo->tm_hour << "_" << timeinfo->tm_min << "_" << timeinfo->tm_sec << "_";
    using namespace std::chrono;
    timestampStream << duration_cast<milliseconds>(start.time_since_epoch()).count();
    return timestampStream.str();
}

void dumpOvTensorInput(const InferenceInput& input, const std::string& dumpDirectoryName) {
    std::stringstream dumpStream;
    std::string fname = std::string("./dump");
    fname = joinPath({fname, getTimestampString(), dumpDirectoryName});
    for (const auto& [name, inputTensor] : input) {
        fname += std::to_string(INPUT_COUNTER++);
        dumpStream << " Name: " << name;
        dumpStream << " Shape: " << inputTensor.get_shape();
        dumpStream << " Type: " << inputTensor.get_element_type();
        dumpStream << "Byte size: " << inputTensor.get_byte_size();
        dumpStream << "Size: " << inputTensor.get_size();
        dumpStream << dumpOvTensor(inputTensor).str();
    }

    std::cout << "Filename: " << fname <<std::endl;
    writeToFile(dumpStream, fname);
}

}  // namespace mediapipe
