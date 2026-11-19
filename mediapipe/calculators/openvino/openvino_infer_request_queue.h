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

#ifndef MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_INFER_REQUEST_QUEUE_H_
#define MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_INFER_REQUEST_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "openvino/openvino.hpp"

namespace mediapipe {
namespace openvino_calculators {

// Lock-light circular queue of idle stream ids, modelled after the OpenVINO
// Model Server `ovms::Queue`. A stream id is an index into `elements_`.
//
// The queue is shared between all graphs/instances that use the same compiled
// model, which is what allows concurrent MediaPipe graphs to reuse a single
// pool of inference requests.
template <typename T>
class Queue {
 public:
  explicit Queue(int length) : ids_(length), front_idx_(0), back_idx_(0) {
    for (int i = 0; i < length; ++i) {
      ids_[i] = i;
    }
  }

  // Blocks until an idle element id is available.
  std::future<int> GetIdleStream() {
    std::promise<int> idle_promise;
    std::future<int> idle_future = idle_promise.get_future();
    std::unique_lock<std::mutex> lk(front_mutex_);
    if (ids_[front_idx_] < 0) {  // Nothing idle, wait for a returned stream.
      std::unique_lock<std::mutex> queue_lock(queue_mutex_);
      promises_.push(std::move(idle_promise));
    } else {
      int value = ids_[front_idx_];
      ids_[front_idx_] = -1;  // Negative value marks a consumed slot.
      front_idx_ = (front_idx_ + 1) % ids_.size();
      lk.unlock();
      idle_promise.set_value(value);
    }
    return idle_future;
  }

  std::optional<int> TryGetIdleStream() {
    std::unique_lock<std::mutex> lk(front_mutex_);
    if (ids_[front_idx_] < 0) {
      return std::nullopt;
    }
    int value = ids_[front_idx_];
    ids_[front_idx_] = -1;
    front_idx_ = (front_idx_ + 1) % ids_.size();
    return value;
  }

  void ReturnStream(int stream_id) {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    if (!promises_.empty()) {
      std::promise<int> promise = std::move(promises_.front());
      promises_.pop();
      lk.unlock();
      promise.set_value(stream_id);
      return;
    }
    lk.unlock();
    std::uint32_t old_back = back_idx_.load();
    while (!back_idx_.compare_exchange_weak(
        old_back, (old_back + 1) % ids_.size(), std::memory_order_relaxed)) {
    }
    ids_[old_back] = stream_id;
  }

  T& GetElement(int stream_id) { return elements_[stream_id]; }

  std::vector<T>& elements() { return elements_; }

  int size() const { return static_cast<int>(ids_.size()); }

 protected:
  std::vector<int> ids_;
  std::uint32_t front_idx_;
  std::atomic<std::uint32_t> back_idx_;
  std::mutex front_mutex_;
  std::mutex queue_mutex_;
  std::vector<T> elements_;
  std::queue<std::promise<int>> promises_;
};

class OVInferRequestsQueue : public Queue<ov::InferRequest> {
 public:
  OVInferRequestsQueue(ov::CompiledModel& compiled_model, int nireq)
      : Queue<ov::InferRequest>(nireq) {
    elements_.reserve(nireq);
    for (int i = 0; i < nireq; ++i) {
      elements_.push_back(compiled_model.create_infer_request());
    }
  }

  ov::InferRequest& GetInferRequest(int stream_id) {
    return GetElement(stream_id);
  }
};

// RAII helper that borrows an inference request from the queue for the
// lifetime of the object and returns it on destruction (also on exceptions or
// early returns from Calculator::Process).
class InferRequestLease {
 public:
  explicit InferRequestLease(OVInferRequestsQueue& queue)
      : queue_(queue), stream_id_(queue.GetIdleStream().get()) {}

  ~InferRequestLease() { queue_.ReturnStream(stream_id_); }

  InferRequestLease(const InferRequestLease&) = delete;
  InferRequestLease& operator=(const InferRequestLease&) = delete;

  int stream_id() const { return stream_id_; }
  ov::InferRequest& infer_request() { return queue_.GetInferRequest(stream_id_); }

 private:
  OVInferRequestsQueue& queue_;
  int stream_id_;
};

}  // namespace openvino_calculators
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_OPENVINO_OPENVINO_INFER_REQUEST_QUEUE_H_
