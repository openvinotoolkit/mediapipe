# Python bindings for the OpenVINO calculators

A pybind11 extension that exposes the MediaPipe graph API together with the
[OpenVINO calculators](../../calculators/openvino/README.md) and a
numpy ↔ `ov::Tensor` packet creator/getter pair.

## Why it is a separate extension

MediaPipe's calculator registry and pybind11's type registry are both
per-shared-object. If the calculators were built into their own `.so` next to
`mediapipe._framework_bindings`, each module would get its own registry copy
and `CalculatorGraph` (living in the other module) would report
`OpenVINOInferenceCalculator` as unregistered.

So `_openvino_bindings.so` links *both* the graph-API submodules from
`//mediapipe/python/pybind:*` and
`//mediapipe/calculators/openvino:openvino_calculators` (whose
`alwayslink = 1` is what runs `REGISTER_CALCULATOR` at load time).

> **Do not `import mediapipe` in the same process.** Both extensions embed
> their own MediaPipe framework, and pybind11 rejects the duplicated
> `Packet` / `CalculatorGraph` / `Timestamp` type registrations.

## Files

| File | Contents |
| --- | --- |
| `openvino_bindings.cc` | `PYBIND11_MODULE(_openvino_bindings, ...)`: graph API submodules + tensor helpers. |
| `__init__.py` | Re-exports the nested pybind submodule members under flat names. |
| `openvino_classification_demo.py` | End-to-end example. |

## API

```python
from mediapipe.python import openvino as mp_ov
```

| Symbol | Notes |
| --- | --- |
| `mp_ov.CalculatorGraph` | Same API as `mediapipe.CalculatorGraph`. |
| `mp_ov.Packet`, `mp_ov.Timestamp`, `mp_ov.ValidatedGraphConfig` | Standard MediaPipe bindings. |
| `mp_ov.create_ov_tensor_vector(list[np.ndarray]) -> Packet` | Wraps `MakePacket<std::vector<ov::Tensor>>`. |
| `mp_ov.get_ov_tensor_vector(Packet) -> list[np.ndarray]` | Reads the payload back. |

The two helpers follow the "custom data type" recipe from
[`python_framework.md`](https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md):

```c++
m.def("create_ov_tensor_vector", [](const std::vector<py::array>& arrays) {
  std::vector<ov::Tensor> tensors;
  for (const auto& a : arrays) tensors.push_back(ArrayToTensor(a));
  return MakePacket<std::vector<ov::Tensor>>(std::move(tensors));
});
```

### dtype mapping

`float16/32/64`, `int8/16/32/64`, `uint8/16/32/64` and `bool` map to the
corresponding `ov::element::Type`. The array shape becomes the tensor shape
one-to-one, so a `[1, 3, 224, 224]` model input needs a 4-D array with an
explicit batch axis. Non C-contiguous arrays are made contiguous first
(`np.ascontiguousarray` is done for you).

Both directions **copy** the buffer. Python objects have independent lifetimes
from MediaPipe packets, and packet payloads must be immutable, so aliasing a
numpy buffer into a packet would be unsafe. For zero-copy pipelines stay in
C++ — see the calculator README.

## Usage

```python
import numpy as np
from mediapipe.python import openvino as mp_ov

config = """
input_stream: "input_tensors"
output_stream: "output_tensors"
node {
  calculator: "OpenVINOSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
      model_path: "/home/user/models/resnet50/resnet50.xml"
      device: "CPU"
      num_infer_requests: 4
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "TENSORS:input_tensors"
  output_stream: "TENSORS:output_tensors"
}
"""

results = []
graph = mp_ov.CalculatorGraph(graph_config=config)
graph.observe_output_stream(
    'output_tensors',
    lambda name, packet: results.append(mp_ov.get_ov_tensor_vector(packet)))
graph.start_run()

frame = np.random.rand(1, 3, 224, 224).astype(np.float32)
graph.add_packet_to_input_stream(
    'input_tensors', mp_ov.create_ov_tensor_vector([frame]).at(0))

graph.close()          # flushes the graph, callbacks have all fired
print(results[0][0].shape)   # (1, 1000)
```

`observe_output_stream` callbacks run on MediaPipe's scheduler threads while
holding the GIL, so keep them short — append and post-process after
`graph.close()`, as the demo does.

Because Python releases the GIL inside `add_packet_to_input_stream`, you can
push packets faster than the model runs; the inference request queue provides
back-pressure once all requests are busy. Raise `num_infer_requests` to let
more frames overlap.

## Demo

```
bazel run -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/python/openvino:openvino_classification_demo -- \
  --model_path=$HOME/models/resnet50/resnet50.xml \
  --input_video_path=$HOME/models/test_video.mp4
```

```
OpenVINOSessionCalculator compiled .../resnet50.xml on CPU with 4 inference requests
frame 0 output[0] shape=(1, 1000) dtype=float32
  top: 419=6.9135 906=6.6229 916=6.2842 549=6.1592 902=5.9980
```

Flags: `--model_path`, `--input_video_path`, `--device`,
`--num_infer_requests`, `--input_width`, `--input_height`, `--max_frames`,
`--top_k`.

The demo depends on `numpy` and `opencv-contrib-python` from
`requirements_lock.txt`, which is why it is run through `bazel run` rather than
the system interpreter.

## Using the module outside Bazel

```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/python/openvino:_openvino_bindings

export PYTHONPATH=$PWD/bazel-bin:$PYTHONPATH
python3 -c "from mediapipe.python import openvino as mp_ov; print(mp_ov.CalculatorGraph)"
```

The interpreter needs `numpy` installed, and `bazel-bin/mediapipe/python` must
contain the `__init__.py` files from the source tree (add `$PWD` to
`PYTHONPATH` as well if you hit import errors).

## Extending the bindings

To expose another custom type, add a creator/getter pair in
`openvino_bindings.cc` next to the existing ones and rebuild — no new shared
object is needed, and keeping everything in one module avoids the registry
duplication described above.
