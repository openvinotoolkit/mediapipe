# OpenVINO GPU classification example

Reads a video through GStreamer VA-API, passes decoded NV12 surfaces to OpenVINO
as owner-retaining remote-frame packets, and runs ResNet-50 inference on the GPU.
The graph performs NV12-to-BGR conversion and resizing in OpenVINO preprocessing;
frames are not copied to host memory on the decoder-to-inference path.

## Model

The graph is configured for an OpenVINO ResNet-50 IR with one static NCHW input
and 1,000 output logits. The input is prepared as `NV12_2:BGR` with
`uint8:fp32` precision and resized to the model's spatial dimensions. Download a
compatible model and provide its XML path with `--model_path`.

## Build

Install the Intel GPU OpenCL runtime, VA-API driver, and GStreamer dependencies
first. The account running the example needs access to the selected DRM render
node. If render-group membership was just enabled with `newgrp render`, restart
the Bazel server from that shell so tests and binaries inherit the group.

```sh
bazel build -c opt \
  --define MEDIAPIPE_ENABLE_GSTREAMER=1 \
  --define MEDIAPIPE_ENABLE_OPENVINO_GPU=1 \
  //mediapipe/examples/desktop/openvino_classification:openvino_classification
```

## Run

`setup_gstreamer.sh` prints the runtime plugin environment variables. Set them
before launching the binary:

```sh
export GST_PLUGIN_PATH=/opt/gstreamer/lib/gstreamer-1.0
export GST_PLUGIN_SCANNER=/opt/gstreamer/bin/gstreamer-1.0/gst-plugin-scanner

bazel-bin/mediapipe/examples/desktop/openvino_classification/openvino_classification \
  --calculator_graph_config_file=mediapipe/examples/desktop/openvino_classification/openvino_classification.pbtxt \
  --model_path="$HOME/models/resnet50/resnet50.xml" \
  --input_video_path="$HOME/models/test_video.mp4" \
  --va_device_path=/dev/dri/renderD128 \
  --max_frames=5
```

`--input_video_path` accepts a local path or a GStreamer URI. `--max_frames=0`
processes the whole video. The graph's output is one logits tensor per frame;
the executable prints its shape, initial values, and top-k class indices.

## Graph path

`GStreamerRemoteTensorCalculator` decodes to `video/x-raw(memory:VAMemory),
format=NV12` using the session's VA display. It emits `REMOTE_FRAME` packets
containing Y and UV remote tensors plus a reference to the originating
GStreamer sample. `OpenVINOInferenceCalculator` binds the planes to the expanded
NV12 model inputs and keeps the sample alive until synchronous inference
finishes. The session uses the same VA display and runs on `GPU`.

The pipeline requires a hardware decoder for the input codec and a working VA
media context. A successful `vainfo` query alone does not guarantee that the
host can create a decoder context; verify the actual GStreamer decode pipeline
on the target machine. A CPU/OpenCV fallback is no longer part of this example.
