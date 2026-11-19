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
//
// Self-contained Python extension exposing the MediaPipe graph API together
// with the OpenVINO calculators and numpy <-> ov::Tensor packet helpers.
//
// The MediaPipe calculator registry and the pybind type registry are per
// shared object, so the graph API bindings have to live in the same module as
// the calculators. Import this module instead of `mediapipe`, not next to it.

#include <cstring>
#include <vector>

#include "mediapipe/framework/packet.h"
#include "mediapipe/python/pybind/calculator_graph.h"
#include "mediapipe/python/pybind/packet.h"
#include "mediapipe/python/pybind/resource_util.h"
#include "mediapipe/python/pybind/timestamp.h"
#include "mediapipe/python/pybind/validated_graph_config.h"
#include "openvino/openvino.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mediapipe {
namespace python {
namespace {

namespace py = pybind11;

ov::element::Type NumpyDtypeToOv(const py::dtype& dtype) {
  if (dtype.kind() == 'f' && dtype.itemsize() == 2) return ov::element::f16;
  const int num = dtype.num();
  if (num == py::dtype::of<float>().num()) return ov::element::f32;
  if (num == py::dtype::of<double>().num()) return ov::element::f64;
  if (num == py::dtype::of<std::int8_t>().num()) return ov::element::i8;
  if (num == py::dtype::of<std::int16_t>().num()) return ov::element::i16;
  if (num == py::dtype::of<std::int32_t>().num()) return ov::element::i32;
  if (num == py::dtype::of<std::int64_t>().num()) return ov::element::i64;
  if (num == py::dtype::of<std::uint8_t>().num()) return ov::element::u8;
  if (num == py::dtype::of<std::uint16_t>().num()) return ov::element::u16;
  if (num == py::dtype::of<std::uint32_t>().num()) return ov::element::u32;
  if (num == py::dtype::of<std::uint64_t>().num()) return ov::element::u64;
  if (num == py::dtype::of<bool>().num()) return ov::element::boolean;
  throw py::value_error("Unsupported numpy dtype for ov::Tensor");
}

py::dtype OvTypeToNumpyDtype(const ov::element::Type& type) {
  if (type == ov::element::f16) return py::dtype("float16");
  if (type == ov::element::f32) return py::dtype::of<float>();
  if (type == ov::element::f64) return py::dtype::of<double>();
  if (type == ov::element::i8) return py::dtype::of<std::int8_t>();
  if (type == ov::element::i16) return py::dtype::of<std::int16_t>();
  if (type == ov::element::i32) return py::dtype::of<std::int32_t>();
  if (type == ov::element::i64) return py::dtype::of<std::int64_t>();
  if (type == ov::element::u8) return py::dtype::of<std::uint8_t>();
  if (type == ov::element::u16) return py::dtype::of<std::uint16_t>();
  if (type == ov::element::u32) return py::dtype::of<std::uint32_t>();
  if (type == ov::element::u64) return py::dtype::of<std::uint64_t>();
  if (type == ov::element::boolean) return py::dtype::of<bool>();
  throw py::value_error("Unsupported ov::Tensor element type: " +
                        type.get_type_name());
}

ov::Tensor ArrayToTensor(const py::array& array) {
  py::array contiguous = py::array::ensure(array, py::array::c_style);
  if (!contiguous) {
    throw py::value_error("Expected a numpy array");
  }
  ov::Shape shape;
  for (py::ssize_t i = 0; i < contiguous.ndim(); ++i) {
    shape.push_back(static_cast<size_t>(contiguous.shape(i)));
  }
  ov::Tensor tensor(NumpyDtypeToOv(contiguous.dtype()), shape);
  std::memcpy(tensor.data(), contiguous.data(), tensor.get_byte_size());
  return tensor;
}

py::array TensorToArray(const ov::Tensor& tensor) {
  std::vector<py::ssize_t> shape(tensor.get_shape().begin(),
                                 tensor.get_shape().end());
  py::array result(OvTypeToNumpyDtype(tensor.get_element_type()), shape);
  std::memcpy(result.mutable_data(), tensor.data(), tensor.get_byte_size());
  return result;
}

}  // namespace

PYBIND11_MODULE(_openvino_bindings, m) {
  m.doc() =
      "MediaPipe graph bindings with the OpenVINO inference/session "
      "calculators linked in.";

  ResourceUtilSubmodule(&m);
  TimestampSubmodule(&m);
  PacketSubmodule(&m);
  CalculatorGraphSubmodule(&m);
  ValidatedGraphConfigSubmodule(&m);

  m.def(
      "create_ov_tensor_vector",
      [](const std::vector<py::array>& arrays) {
        std::vector<ov::Tensor> tensors;
        tensors.reserve(arrays.size());
        for (const auto& array : arrays) {
          tensors.push_back(ArrayToTensor(array));
        }
        return MakePacket<std::vector<ov::Tensor>>(std::move(tensors));
      },
      py::arg("arrays"),
      "Creates a Packet holding std::vector<ov::Tensor> from numpy arrays.");

  m.def(
      "get_ov_tensor_vector",
      [](const Packet& packet) {
        if (!packet.ValidateAsType<std::vector<ov::Tensor>>().ok()) {
          throw py::type_error(
              "Packet does not hold std::vector<ov::Tensor>.");
        }
        std::vector<py::array> arrays;
        for (const ov::Tensor& tensor :
             packet.Get<std::vector<ov::Tensor>>()) {
          arrays.push_back(TensorToArray(tensor));
        }
        return arrays;
      },
      py::arg("packet"),
      "Returns the ov::Tensor payload of a Packet as a list of numpy arrays.");
}

}  // namespace python
}  // namespace mediapipe
