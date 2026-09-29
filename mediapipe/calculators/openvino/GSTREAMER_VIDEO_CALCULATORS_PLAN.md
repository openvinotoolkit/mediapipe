# GStreamer Video and OpenVINO RemoteTensor Plan 

 

## Goal 

 

Add GStreamer-backed video source and encoder calculators that build and run on 

Linux and Windows. The primary zero-copy workflow is video decode to OpenVINO 

inference: the source calculator emits device-backed `ov::RemoteTensor` 

packets, and `OpenVINOInferenceCalculator` consumes them without a host-memory 

round trip. The existing OpenVINO classification example will exercise this 

workflow. 

 

This document is an implementation plan, not a claim that the platform 

interoperability or build has already been validated. 

 

## Existing Components 

 

- [OpenVINO calculator README](README.md) documents the `REMOTE_TENSORS` input 

  as `std::vector<ov::RemoteTensor>`. 

- [OpenVINOInferenceCalculator](openvino_inference_calculator.cc) binds one 

  vector element to each model input with `set_tensor()` or 

  `set_input_tensor()`. 

- [OpenVINO session calculator](openvino_session_calculator.cc) supports 

  `NV12_2` preprocessing configuration, but the inference calculator does not 

  currently bind multiple remote tensors to one model input. 

- [OpenVINO classification example](../../examples/desktop/openvino_classification/README.md) 

  already reads video, but currently uses OpenCV and creates host tensors. 

- `rules_foreign_cc` is already present in `WORKSPACE`. The OpenVINO binary 

  dependency currently describes Linux libraries and excludes the GPU plugin 

  from the normal target. 

- The external Intel MPI reference demonstrates GStreamer pipelines and 

  Windows D3D11 use. Its video decode path copies frames through CPU memory, so 

  it is not evidence of a zero-copy implementation. 

 

## Packet and Zero-Copy Contract 

 

Use the existing `REMOTE_TENSORS` stream wherever its payload and lifetime 

semantics are sufficient. The protobuf file defines calculator options; the 

actual stream payload is the C++ vector type. MediaPipe packet timestamps carry 

the frame presentation time, so a separate timestamp field is not initially 

needed. 

 

The current inference binding assumes one tensor per model input. For an NV12 

model input represented by separate Y and UV remote tensors, extend the 

inference binding to group multiple tensors for a single input and use the 

OpenVINO multi-tensor input API. Keep the existing one-tensor-per-input path 

unchanged. Confirm this against the supported OpenVINO API and preprocessing 

configuration before implementation. 

 

A `RemoteTensor` packet must keep the backing video surface alive until 

inference and all packet consumers have finished. Verify whether each 

platform's OpenVINO remote tensor retains the imported resource. If it does 

not, define a small packet-owner wrapper that holds the remote-tensor vector 

and a referenced GStreamer buffer; adapt the inference calculator to read the 

tensors from that wrapper. Do not release or recycle the decoder surface when 

the source calculator returns from `Process()`. 

 

The zero-copy guarantee applies to decoder packet to inference input on a 

supported device path. The inference calculator currently copies outputs out 

of its reusable infer-request buffers into host `ov::Tensor`s; classification 

logits can continue to use that behavior. Avoid silently falling back to CPU 

conversion for the decoder-to-inference path: unsupported memory types or 

contexts should produce a clear error or use an explicitly selected fallback. 

 

## Cross-Platform Design 

 

Keep calculator options and graph-level behavior shared. Put GStreamer memory 

interop, OpenVINO remote-context setup, runtime library loading, and any native 

surface ownership in platform-specific implementations. 

 

| Platform | Intended video memory path | OpenVINO interop to validate | 

| --- | --- | --- | 

| Linux | GStreamer VA-API elements and `memory:VAMemory` | VA display sharing and `ov::RemoteTensor` creation from the decoded VA surface | 

| Windows | GStreamer D3D11 elements and `memory:D3D11Memory` | D3D11 device sharing and `ov::RemoteTensor` creation from the decoded D3D11 surface | 

 

Before implementing the calculators, verify that the selected GStreamer 

release, plugins, OpenVINO GPU plugin, and remote-context APIs are available 

for both supported build platforms. The current Linux-only OpenVINO repository 

needs a Windows counterpart or platform-specific repository selection. Do 

not assume a VA-API context or Linux plugin dependency applies on Windows. 

 

## Implementation Steps 

 

### 1. Validate platform prerequisites 

 

- Pin a GStreamer release, initially 1.28.2 as used by the reference build. 

- Verify Meson and `rules_foreign_cc` can build the required GStreamer 

  subprojects on Linux and Windows with the repository's Bazel toolchains. 

- Enumerate native prerequisites for each platform, including GLib, libva and 

  libdrm on Linux and the required Windows SDK/D3D11 dependencies on Windows. 

- Verify hardware decode/encode elements, OpenVINO remote contexts, resource 

  ownership, and supported NV12 tensor layouts on actual target systems. 

- If a platform cannot build from source through the selected Bazel path, 

  resolve that build constraint before promising calculator support there. 

 

### 2. Integrate GStreamer into Bazel 

 

- Add pinned external GStreamer source repositories and Bazel targets using 

  `rules_foreign_cc` Meson integration. 

- Build the core, app and video libraries plus only the plugins required for 

  source/decode, hardware conversion, encode, parsers, muxers, and the example. 

- Disable tests, examples, and unused optional plugins to keep the build 

  focused; prevent Meson from downloading undeclared subprojects during a 

  sandboxed build. 

- Provide platform-specific native dependency targets and compile/link 

  settings. Keep Linux and Windows build selections explicit. 

- Make shared libraries, GStreamer plugins, and the plugin scanner available 

  through Bazel runfiles at execution time. Ensure plugin discovery does not 

  accidentally use an unrelated system GStreamer installation. 

- Add or select platform-specific OpenVINO archives and GPU plugin targets; 

  preserve the existing CPU-only build. 

 

### 3. Share OpenVINO and GStreamer device contexts 

 

- Add a session-owned platform context/display holder so the model and 

  GStreamer refer to the same native device context where required. 

- On Linux, investigate sharing a VA display and wrapping VA surfaces through 

  OpenVINO's supported GPU remote-context API. let GStreamer create the surface, then pull GStreamer's own 

  native VA display back out of the decoded buffer 

and give that to OpenVINO's `VAContext`. Same display 

 

- On Windows, investigate sharing the D3D11 device/surface through the 

  supported OpenVINO GPU remote-context API. 

- Keep native handles alive for at least as long as the session and every 

  in-flight tensor that depends on them. 

- Make device selection configurable and report context mismatch or unavailable 

  hardware as a useful calculator error. 

 

### 4. Add the GStreamer video source calculator 

 

- Add a calculator that accepts a file, URL/stream, or camera source and emits 

  timestamped `REMOTE_TENSORS` packets. 

- Add options for source, output dimensions, pixel format, buffering, and an 

  optional pipeline override only if required by supported sources. 

- Use Linux VA-API memory and Windows D3D11 memory paths, with platform-specific 

  pipeline elements selected in the implementation. 

- Wrap decoded surfaces as OpenVINO remote tensors without mapping them to host 

  memory on the zero-copy path. 

- Retain the source buffer or surface owner until downstream inference releases 

  the packet. 

- Handle EOS, negotiation failures, bus errors, stream timestamps, and orderly 

  pipeline shutdown. 

 

### 5. Connect remote video frames to inference 

 

- Extend `OpenVINOInferenceCalculator` to bind NV12 plane groups to one model 

  input when required, while preserving current `REMOTE_TENSORS` behavior for 

  ordinary one-tensor inputs. 

- Make input ordering and plane grouping explicit and validate them against 

  the session model and preprocessing options. 

- Keep the producer's frame packet alive through synchronous inference; ensure 

  packet fan-out or queued work also retains the surface correctly. 

- Verify that the device tensors use the same session context as the compiled 

  model. Reject incompatible contexts rather than silently copying to CPU. 

 

### 6. Add the GStreamer video encoder calculator 

 

- Add an encoder calculator with codec, container, frame size, frame rate, 

  bitrate, and output destination options, following the shape of 

  [OpenCvVideoEncoderCalculatorOptions](../video/opencv_video_encoder_calculator.proto). 

- Accept remote video tensors or the frame-owner packet, depending on the 

  lifetime requirements established in Step 3. 

- Use the native surface fast path when the input is a compatible GStreamer 

  buffer. Make any tensor-to-surface copy or format conversion explicit. 

- Finalize the pipeline on close by sending EOS and waiting for success or a 

  bus error. 

 

### 7. Extend the classification example 

 

- Extend the existing `openvino_classification` application and graph rather 

  than creating a separate basic example. 

- Replace the OpenCV frame-read and host `FrameToTensor` ingress with the 

  GStreamer source calculator feeding `REMOTE_TENSORS` into 

  `OpenVINOInferenceCalculator`. 

- Keep classification output handling and top-k reporting. Document the model 

  input format, device selection, required media runtime, and Bazel build/run 

  commands for both Linux and Windows. 

- Preserve a CPU/host-tensor mode if useful for machines without supported 

  hardware; label it as a fallback, not as the zero-copy path. 

 

## Tests and Acceptance Criteria 

 

- **Cross-platform build checks:** build the GStreamer dependency, calculator 

  libraries, and classification example on Linux and Windows. 

- **CPU tests:** test option validation, packet/timestamp handling, EOS, error 

  propagation, and calculator lifecycle without requiring a GPU. 

- **Hardware integration tests:** on Linux and Windows separately, decode a 

  small fixture and run inference using the platform remote context. Gate tests 

  on the matching device/runtime being present. 

- **Zero-copy verification:** assert the emitted tensors are remote tensors 

  backed by the expected device/context; verify that their surface owner stays 

  alive through inference. Use instrumentation or a focused test hook to 

  ensure the zero-copy path does not map/copy frames through host memory. 

- **NV12 binding tests:** test one model input represented by the required 

  plane group, input ordering, invalid plane counts, and incompatible model 

  layouts. 

- **Classification example test:** run a short video through decode and 

  classification and verify frame count, timestamps, and output tensor shape. 

- **Encoder integration tests:** encode a short compatible stream, finalize it, 

  and verify the resulting media is readable. Cover both platform hardware 

  paths where available. 

- Keep hardware tests skippable with a clear reason when a runner lacks the 

  necessary device or driver; do not treat a skipped hardware test as proof of 

  zero-copy behavior. 

 

Acceptance requires successful Linux and Windows builds, a functioning 

classification workflow, and hardware-backed zero-copy verification on each 

platform claimed as supported. CPU fallback behavior must be separately 

documented and tested. 

 

## Main Risks 

 

- Meson foreign builds may have different sandboxing, dependency-discovery, or 

  toolchain requirements across Linux and Windows. 

- GStreamer plugins and OpenVINO remote-surface APIs may not expose equivalent 

  memory paths on the two platforms or on all supported hardware. 

- A remote tensor may not retain the GStreamer buffer or decoder-pool surface; 

  surface lifetime must be proven, not inferred from the tensor handle alone. 

- The current inference calculator's one-tensor-per-input mapping is 

  insufficient for a two-plane NV12 input bound to one model input. 

- Runtime plugin and shared-library discovery must work from Bazel runfiles on 

  both operating systems. 