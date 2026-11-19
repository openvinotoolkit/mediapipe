# OpenVINO calculators

A pair of MediaPipe calculators that run inference with the
[OpenVINO Runtime](https://docs.openvino.ai/):

| Calculator | Role |
| --- | --- |
| `OpenVINOSessionCalculator` | Compiles a model once and publishes a shared pool of `ov::InferRequest` objects as an output side packet. |
| `OpenVINOInferenceCalculator` | Borrows a free request from that pool for every incoming packet and supports vector or named tensor streams. |

Model compilation is a graph-lifetime operation, while inference is a
per-packet operation. Keeping them separate lets graphs share one compiled
model and request pool. In OVMS, this interface is used with direct model paths
and does not select servables from the Model Server model configuration.

## Files

| File | Contents |
| --- | --- |
| `openvino_session_calculator.cc/.proto` | Model compilation + session registry. |
| `openvino_inference_calculator.cc/.proto` | Per-packet inference. |
| `openvino_session.h` | `OpenVINOSession` — the type carried by the `SESSION` side packet. |
| `openvino_infer_request_queue.h` | `Queue<T>`, `OVInferRequestsQueue`, `InferRequestLease`. |

## How it works

```
                       ┌───────────────────────────┐
                       │ OpenVINOSessionCalculator │
                       │  ov::Core::compile_model  │
                       │  N × create_infer_request │
                       └────────────┬──────────────┘
                        SESSION side packet
                       (shared_ptr<OpenVINOSession>)
                                    │
  TENSORS ───────► ┌────────────────┴─────────────────┐ ───────► TENSORS
  (vector<Tensor>) │   OpenVINOInferenceCalculator    │   (vector<ov::Tensor>)
                   │  lease = queue.GetIdleStream()   │
                   │  set_input_tensor / infer()      │
                   │  queue.ReturnStream(lease)       │
                   └──────────────────────────────────┘
```

`OpenVINOSession` owns the `ov::CompiledModel` and an `OVInferRequestsQueue`.
The queue is the OVMS `ovms::Queue` design: a circular buffer of idle request
indices plus a `std::queue<std::promise<int>>`. When a request is free,
`GetIdleStream()` returns a ready future immediately; when the pool is
exhausted, the caller's promise is parked and fulfilled by whichever
`ReturnStream()` happens next. This means `Process()` blocks (rather than
spinning or failing) when all requests are busy, and MediaPipe's scheduler keeps
the other graph nodes running on their own threads meanwhile.

`InferRequestLease` is an RAII wrapper: it acquires an id in its constructor and
calls `ReturnStream()` in its destructor, so a request is returned even if
inference throws or `Process()` returns early with an error.

## `OpenVINOSessionCalculator`

Node parameters (`OpenVINOSessionCalculatorOptions`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model_path` | string | *required* | `.xml` (OpenVINO IR), `.onnx`, `.pb`, `.tflite`, `.pdmodel` — anything `ov::Core::compile_model` accepts. |
| `device` | string | `"CPU"` | `CPU`, `GPU`, `GPU.1`, `NPU`, `AUTO:GPU,CPU`, `MULTI:CPU,GPU`, `HETERO:GPU,CPU`, `BATCH:GPU`. |
| `plugin_config` | repeated `{key, value}` | empty | Passed verbatim to `compile_model` as `ov::AnyMap`. |
| `num_infer_requests` | uint32 | `0` | Size of the request queue. `0` = `ov::optimal_number_of_infer_requests` reported by the plugin. |
| `shape` | string | empty | `auto`, a tuple such as `(1,3,224,224)`, or a JSON input map such as `{"input1":"(1,10)","input2":"auto"}`. Takes precedence over `batch_size`. |
| `batch_size` | string | empty | A positive integer or `auto`. The first dimension is used when the model has no batch layout metadata. |
| `layout` | string | empty | A layout such as `NCHW`, a conversion such as `NHWC:NCHW`, or a JSON map of input/output names to layouts. |
| `mean` | string | empty | A scalar, tuple, or array subtracted from a single model input. |
| `scale` | string | empty | A scalar, tuple, or array used to divide a single model input. |
| `color_format` | string | empty | Incoming color format or conversion such as `RGB:BGR`. Supports `RGB`, `BGR`, `GRAY`, `NV12`, `NV12_2`, `I420`, and `I420_3`. |
| `precision` | string | empty | Incoming precision or conversion such as `uint8:fp32`. Supports OVMS precision names including `fp32`, `uint8`, and `bf16`. |

Contract: no input streams, no output streams, one output side packet
`SESSION` of type `std::shared_ptr<OpenVINOSession>`.

```pbtxt
node {
  calculator: "OpenVINOSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
      model_path: "/models/resnet50/resnet50.xml"
      device: "CPU"
      plugin_config { key: "PERFORMANCE_HINT" value: "THROUGHPUT" }
      plugin_config { key: "NUM_STREAMS" value: "4" }
      num_infer_requests: 8
      shape: "(1,224,224,3)"
      layout: "NHWC:NCHW"
      mean: "[123.675,116.28,103.53]"
      scale: "[58.395,57.12,57.375]"
      color_format: "RGB:BGR"
      precision: "uint8:fp32"
    }
  }
}
```

A colon-separated value describes the incoming tensor first and the model-side
representation second. Model preparation happens before compilation using
OpenVINO's `PrePostProcessor`. Unnamed `shape` and `layout` values require a
single-input model; JSON shape/layout maps support multi-input models, and
layout maps can also name outputs. Mean, scale, color, and precision currently
apply to single-input models, matching the corresponding OVMS preprocessing
path.

`shape: "auto"` and `batch_size: "auto"` compile a ranked dynamic model, so a
single graph session can accept compatible runtime dimensions supported by the
selected OpenVINO plugin. Unlike OpenVINO Model Server, this calculator does
not reload the model for each new request shape because the session is created
before MediaPipe input packets arrive.

A single process-wide `ov::Core` is used, so the plugin-level model cache
(`CACHE_DIR`) and device initialisation are shared.

## `OpenVINOInferenceCalculator`

`SESSION` is always an input side packet of type
`std::shared_ptr<OpenVINOSession>`. For tensor streams, choose one of two
mutually exclusive contracts:

| Mode | Inputs | Outputs |
| --- | --- | --- |
| Vector | One `TENSORS` stream carrying `std::vector<ov::Tensor>`, or one `REMOTE_TENSORS` stream carrying `std::vector<ov::RemoteTensor>`. Vector length must equal the model input count. | One `TENSORS` stream carrying `std::vector<ov::Tensor>`. |
| Named | One tagged `ov::Tensor` stream per model input, mapped to model names. | One tagged `ov::Tensor` stream per requested output, mapped from model names. |

Do not mix vector and named streams on the same side of one inference node.
Named mode requires a distinct tag for each stream, one binding for every model
input, and one binding for each output stream. It can connect branch outputs
directly without a merge/split adapter:

```pbtxt
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "INPUT_A:branch_a"
  input_stream: "INPUT_B:branch_b"
  output_stream: "SCORES:scores"
  output_stream: "EMBEDDING:embedding"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOInferenceCalculatorOptions] {
      tag_to_input_tensor_names { key: "INPUT_A" value: "input_a" }
      tag_to_input_tensor_names { key: "INPUT_B" value: "input_b" }
      tag_to_output_tensor_names { key: "SCORES" value: "scores" }
      tag_to_output_tensor_names { key: "EMBEDDING" value: "embedding" }
    }
  }
}
```

Vector mode remains more compact when a producer naturally emits one tensor
vector or when graph branches are not important. `input_order` maps vector
positions to model input names; when omitted, binding is by model index.
`output_order` selects and orders output tensors; when omitted, model output
order is used. Do not set order options together with their corresponding named
stream mapping.

```pbtxt
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "TENSORS:input_tensors"
  output_stream: "TENSORS:output_tensors"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOInferenceCalculatorOptions] {
      input_order: "images"
      output_order: "logits"
    }
  }
}
```

### Preparing the input

Without session preprocessing options, the tensor must match the model's
element type and shape. With preprocessing configured, it must instead match
the incoming precision, shape, layout, and color format declared in the
session options. Typical producer code:

```c++
ov::Tensor tensor(ov::element::f32, ov::Shape{1, 3, 224, 224});
// ... fill tensor.data<float>() ...
auto packet_payload = std::make_unique<std::vector<ov::Tensor>>();
packet_payload->push_back(std::move(tensor));
graph.AddPacketToInputStream(
    "input_tensors", mediapipe::Adopt(packet_payload.release()).At(ts));
```

If you prefer the model to accept raw NHWC `u8` images, bake the conversion
into the model with `ov::preprocess::PrePostProcessor` before saving it, or
reshape/convert it in the node that feeds this calculator.

## Zero-copy between nodes

`ov::Tensor` is a thin handle around a `std::shared_ptr<ov::ITensor>`. Copying
an `ov::Tensor` copies the handle, not the buffer. A MediaPipe packet holds an
immutable, reference-counted payload, and downstream calculators read it with
`Get<std::vector<ov::Tensor>>()`, which returns a `const&`. So:

* **Producer → inference node**: the buffer you allocated is handed to
  `ov::InferRequest::set_input_tensor()` as-is. OpenVINO only copies internally
  if the buffer is unsuitable (wrong precision, or not aligned for the plugin);
  allocate with `ov::Tensor(type, shape)` so that the plugin's allocator is
  used and the data pointer stays live.
* **Inference node → consumer**: the output vector is moved into the packet,
  and named output packets contain `ov::Tensor` handles. Downstream fan-out
  does not copy tensor buffers.

One copy does remain today, inside `Process()`, for both vector and named
outputs:

```c++
ov::Tensor copy(source.get_element_type(), source.get_shape());
source.copy_to(copy);
```

It is required because the `ov::InferRequest` goes straight back into the
shared queue and its internal output buffer will be overwritten by the next
packet. A future zero-copy output mode must keep the inference request leased
until downstream packets release their references; otherwise a later inference
can overwrite data still being read. Pre-allocating output tensors and binding
them before `infer()` avoids this copy only for static output shapes, but does
not by itself tie request reuse to downstream packet lifetime:

```c++
ov::Tensor out(port.get_element_type(), port.get_shape());  // static shapes only
request.set_output_tensor(i, out);
request.infer();
outputs->push_back(std::move(out));  // no copy, buffer is owned by the packet
```

This is only safe for models with fully static output shapes, and it opts out
of some plugin-side output buffer reuse, which is why it is not the default.

## Using `ov::RemoteTensor` (GPU / NPU, zero host round-trip)

`ov::RemoteTensor` derives from `ov::Tensor` and wraps a device-side buffer
(an OpenCL `cl_mem` / VA surface / DirectX surface). Feeding it to the
inference node keeps the data on the device across the whole pipeline — no
host copy on the way in, and inference reads the surface directly.

1. Connect the `REMOTE_TENSORS` tag instead of `TENSORS`:

```pbtxt
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "REMOTE_TENSORS:gpu_tensors"
  output_stream: "TENSORS:output_tensors"
}
```

2. Compile the model on a device that supports remote contexts:

```pbtxt
node_options: {
  [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
    model_path: "/models/resnet50/resnet50.xml"
    device: "GPU"
  }
}
```

3. Produce the tensors from the *same* context the model was compiled with —
   a remote tensor from a different context is rejected by the plugin:

```c++
auto session = side_packet.Get<OpenVINOSessionPtr>();
ov::RemoteContext ctx = session->compiled_model.get_context();

// Allocate a device buffer...
auto gpu_ctx = ctx.as<ov::intel_gpu::ocl::ClContext>();
ov::RemoteTensor remote = gpu_ctx.create_tensor(
    ov::element::f32, ov::Shape{1, 3, 224, 224});

// ...or wrap a cl_mem / VASurfaceID you already own (true zero copy from a
// GPU decoder or an OpenCL preprocessing kernel):
ov::RemoteTensor wrapped = gpu_ctx.create_tensor(ov::element::u8, shape, cl_buf);

auto payload = std::make_unique<std::vector<ov::RemoteTensor>>();
payload->push_back(std::move(wrapped));
```

Inside `Process()` the remote tensors are put into a `std::vector<ov::Tensor>`;
this is a handle copy that keeps the device impl pointer, so the buffer is
never brought to host memory.

Outputs are currently always emitted as host `ov::Tensor`. To keep results on
the device as well, bind a remote output tensor with `set_output_tensor()`
using the pre-allocation pattern shown above.

Building for GPU: `third_party/openvino.BUILD` excludes
`libopenvino_intel_gpu_plugin.so` from linking because it needs
`libOpenCL.so.1`, which is not part of the archive. Install an OpenCL ICD
loader (`ocl-icd-libopencl1` + the Intel compute runtime) and move the plugin
from the `openvino_gpu_plugin` filegroup into the `openvino` target's `srcs`.

## Queue size

The queue size is the number of packets that can be in flight in this node at
the same time.

* `num_infer_requests: N` — fixed pool of `N` requests.
* `num_infer_requests: 0` (or unset) — the plugin decides via
  `ov::optimal_number_of_infer_requests`. With
  `PERFORMANCE_HINT=THROUGHPUT` on a large CPU this can be well over a hundred;
  with `PERFORMANCE_HINT=LATENCY` it is typically 1.

Useful combinations:

```pbtxt
# Lowest latency, one request at a time.
plugin_config { key: "PERFORMANCE_HINT" value: "LATENCY" }
num_infer_requests: 1

# Throughput, bounded pool.
plugin_config { key: "PERFORMANCE_HINT" value: "THROUGHPUT" }
num_infer_requests: 8

# Explicit CPU stream control.
plugin_config { key: "NUM_STREAMS" value: "4" }
plugin_config { key: "INFERENCE_NUM_THREADS" value: "16" }
num_infer_requests: 4
```

Rules of thumb: set `num_infer_requests` ≥ the number of `NUM_STREAMS` (or the
number of graphs feeding the node) to keep the device busy, and keep it bounded
so that memory use and tail latency stay predictable. Requests are allocated
eagerly in `OpenVINOSession`'s constructor, so a large pool costs memory from
the start. Because acquisition blocks, the queue also acts as a natural
back-pressure valve for the whole graph.

## Sharing one model between concurrent graphs

`OpenVINOSessionCalculator::Open()` looks the session up in a process-wide
registry keyed by `model_path | device | num_infer_requests | plugin_config`.
On a hit it publishes the existing `shared_ptr`; on a miss it compiles the
model and inserts a `weak_ptr`. Consequences:

* **Identical option blocks share everything** — one `compile_model()` call,
  one set of device streams, one request queue. Two graphs each sending a
  packet will take two different requests out of the same pool.
* **Different option blocks stay separate.** Changing a single
  `plugin_config` entry, the device, or `num_infer_requests` produces a
  different key and therefore a second compiled model.
* **Lifetime is reference-counted.** The registry only holds `weak_ptr`s, so
  the compiled model is released when the last graph using it is destroyed,
  and recompiled on demand afterwards.
* Compilation is serialised by a mutex, so N graphs starting simultaneously
  compile the model once, not N times.

```c++
// Same config text for both graphs -> one compiled model, one shared queue.
mediapipe::CalculatorGraph graph_a, graph_b;
graph_a.Initialize(config);
graph_b.Initialize(config);
graph_a.StartRun({});
graph_b.StartRun({});
```

The same applies within one graph: two `OpenVINOSessionCalculator` nodes with
equal options resolve to the same session, which is handy when different
branches of a graph need the same model.

If you want two graphs to be *isolated* (separate queues, separate device
streams) while still using the same file, give them different option blocks,
for example by setting different `num_infer_requests` values or adding a
distinguishing `plugin_config` entry.

## Build

The calculators link against the OpenVINO Runtime binary package declared in
`WORKSPACE` as `@openvino` (see `third_party/openvino.BUILD`).

```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/openvino:openvino_calculators
```

Depend on `//mediapipe/calculators/openvino:openvino_calculators` from your
binary (both calculator libraries are `alwayslink = 1`, which is what gets
`REGISTER_CALCULATOR` to run).

## Testing

`openvino_calculators_concurrency_test.cc` builds 4 `CalculatorGraph`
instances around the same `add.xml`/`add.bin` model (in `testdata/`, an
OpenVINO IR that sums two `f32[1,10]` inputs), runs them concurrently on
separate threads with distinct inputs, and checks each output against the
expected sum. Since the 4 graphs share identical session options, they also
share one compiled model and one 4-request queue (see "Sharing one model
between concurrent graphs" above), so the test doubles as a concurrency check
for `OVInferRequestsQueue`/`InferRequestLease`.

```
bazel test -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/calculators/openvino:openvino_calculators_concurrency_test \
  --test_output=all
```

## See also

* `mediapipe/examples/desktop/openvino_classification` — C++ example.
* `mediapipe/python/openvino` — Python bindings and example.
