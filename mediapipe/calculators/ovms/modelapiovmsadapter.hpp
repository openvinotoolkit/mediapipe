#pragma once
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
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <adapters/inference_adapter.h>  // model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT

#ifdef _WIN32
#include <Windows.h>
#endif

// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib

class OVMS_Server_;
typedef struct OVMS_Server_ OVMS_Server;

namespace mediapipe::ovms {

using InferenceOutput = std::map<std::string, ov::Tensor>;
using InferenceInput = std::map<std::string, ov::Tensor>;

using shape_border_t = std::vector<int64_t>;
using shape_min_max_t = std::pair<shape_border_t, shape_border_t>;
using shapes_min_max_t = std::unordered_map<std::string, shape_min_max_t>;
class OVMSInferenceAdapter : public ::InferenceAdapter {
    OVMS_Server* cserver{nullptr};
    const std::string servableName;
    uint32_t servableVersion;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    shapes_min_max_t inShapesMinMaxes;
    shapes_min_max_t outShapesMinMaxes;
    std::unordered_map<std::string, ov::element::Type_t> inputDatatypes;
    std::unordered_map<std::string, ov::element::Type_t> outputDatatypes;
    ov::AnyMap modelConfig;

    // Creates (once per process) and returns a shared OVMS server handle for adapters
    // that are constructed without an explicit server pointer.
    // This path is used by calculators running in-process when they rely on the default
    // OVMS singleton instead of receiving a handle from runtime/shared-library plumbing.
    // Keeping one shared handle avoids repeated OVMS_ServerNew calls and keeps all such
    // adapters bound to the same server instance.
    static OVMS_Server* getSharedServerHandle() {
        static std::once_flag once;
        static OVMS_Server* sharedServer{nullptr};
        std::fprintf(stderr, "OVMSAdapter shared handle call_once entry thread=%zu\n", std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::call_once(once, []() {
            std::fprintf(stderr, "OVMSAdapter shared handle init start\n");
            auto* serverNewFn = &OVMS_ServerNew;
            std::fprintf(stderr, "OVMSAdapter shared handle OVMS_ServerNew fn=%p\n", reinterpret_cast<void*>(serverNewFn));
#ifdef _WIN32
            HMODULE mod = nullptr;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(serverNewFn), &mod) != 0) {
                char modulePath[MAX_PATH] = {0};
                DWORD pathLen = GetModuleFileNameA(mod, modulePath, MAX_PATH);
                if (pathLen > 0) {
                    std::fprintf(stderr, "OVMSAdapter shared handle OVMS_ServerNew module=%s\n", modulePath);
                }
            }
#endif
            OVMS_Status* status = serverNewFn(&sharedServer);
            std::fprintf(stderr, "OVMSAdapter shared handle init OVMS_ServerNew returned status=%p server=%p\n", static_cast<void*>(status), static_cast<void*>(sharedServer));
            if (status != nullptr) {
                const char* msg = nullptr;
                OVMS_StatusDetails(status, &msg);
                std::string details = (msg != nullptr) ? msg : "unknown error";
                OVMS_StatusDelete(status);
                throw std::runtime_error("OVMS_ServerNew failed in OVMSInferenceAdapter: " + details);
            }
        });
        std::fprintf(stderr, "OVMSAdapter shared handle call_once exit server=%p\n", static_cast<void*>(sharedServer));
        return sharedServer;
    }

public:
    // TODO Windows: Fix definition in header - does not compile in cpp.
    OVMSInferenceAdapter(const std::string& servableName, uint32_t servableVersion = 0, OVMS_Server* server = nullptr) :
        servableName(servableName),
        servableVersion(servableVersion) {
        if (nullptr != server) {
            this->cserver = server;
        } else {
            this->cserver = getSharedServerHandle();
        }
    }
    virtual ~OVMSInferenceAdapter();
    InferenceOutput infer(const InferenceInput& input) override;
    void infer(const InferenceInput& input, InferenceOutput& output) override;
    void loadModel(const std::shared_ptr<const ov::Model>& model, ov::Core& core,
        const std::string& device, const ov::AnyMap& compilationConfig, size_t max_num_requests = 1) override;
    void inferAsync(const InferenceInput& input, const CallbackData callback_args) override;
    void setCallback(std::function<void(ov::InferRequest, const CallbackData)> callback);
    bool isReady();
    void awaitAll();
    void awaitAny();
    size_t getNumAsyncExecutors() const;
    ov::PartialShape getInputShape(const std::string& inputName) const override;
    ov::PartialShape getOutputShape(const std::string& outputName) const override;
    ov::element::Type_t getInputDatatype(const std::string& inputName) const override;
    ov::element::Type_t getOutputDatatype(const std::string& outputName) const override;
    std::vector<std::string> getInputNames() const override;
    std::vector<std::string> getOutputNames() const override;
    const ov::AnyMap& getModelConfig() const override;
};
}  // namespace mediapipe
