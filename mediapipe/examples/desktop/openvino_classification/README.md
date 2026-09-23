# OpenVINO classification example (C++)

Reads frames from a video file with OpenCV, runs image classification on every
frame through a MediaPipe graph built from the
[OpenVINO calculators](../../../calculators/openvino/README.md), and prints the
output tensor.

## Files

| File | Contents |
| --- | --- |
| `openvino_classification_main.cc` | Video reading, frame → `ov::Tensor` conversion, graph driving, tensor printing. |
| `openvino_classification.pbtxt` | Two-node graph: session + inference. |

## Get a model

```
mkdir -p ~/models/resnet50 && cd ~/models/resnet50
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -O
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -O
```

Input is `[1, 3, 224, 224]` fp32 NCHW, output is `[1, 1000]` logits. Point
`model_path` in `openvino_classification.pbtxt` at the downloaded `.xml`.

A test video can be generated with:

```
ffmpeg -f lavfi -i testsrc=size=640x480:rate=10:duration=2 -pix_fmt yuv420p ~/models/test_video.mp4
```

## Build and run

```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/examples/desktop/openvino_classification:openvino_classification

bazel-bin/mediapipe/examples/desktop/openvino_classification/openvino_classification \
  --calculator_graph_config_file=mediapipe/examples/desktop/openvino_classification/openvino_classification.pbtxt \
  --input_video_path=$HOME/models/test_video.mp4 \
  --max_frames=5
```

```
OpenVINOSessionCalculator compiled .../resnet50.xml on CPU with 4 inference requests
frame 0 output[0] shape=[1,1000] type=f32
  values: -2.39221 1.0669 -3.18584 ...
  top5: 419=6.91352 906=6.62288 916=6.28421 549=6.15924 902=5.99802
```

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--calculator_graph_config_file` | *required* | Text-format `CalculatorGraphConfig`. |
| `--input_video_path` | *required* | Anything `cv::VideoCapture` opens. |
| `--input_width` / `--input_height` | 224 | Target size of the resized frame. |
| `--input_nhwc` | false | Emit `{1,H,W,3}` instead of `{1,3,H,W}`. |
| `--swap_rb` | true | Convert the OpenCV BGR frame to RGB. |
| `--scale` | 1.0 | Per-pixel multiplier, e.g. `0.00392156862` for `[0,1]`. |
| `--max_frames` | 0 | Stop after N frames; 0 = whole video. |
| `--top_k` | 5 | Number of top scoring classes printed per frame. |

## How the main loop works

```c++
graph.Initialize(config);
auto poller = graph.AddOutputStreamPoller("output_tensors");
graph.StartRun({});

while (capture.read(frame)) {
  ov::Tensor tensor = FrameToTensor(frame, ...);         // cv::Mat -> ov::Tensor
  auto payload = std::make_unique<std::vector<ov::Tensor>>();
  payload->push_back(std::move(tensor));
  graph.AddPacketToInputStream(
      "input_tensors", mediapipe::Adopt(payload.release()).At(Timestamp(i)));

  mediapipe::Packet out;
  poller.Next(&out);                                     // blocking, 1:1 with input
  PrintTensor(i, out.Get<std::vector<ov::Tensor>>(), top_k);
}
graph.CloseInputStream("input_tensors");
graph.WaitUntilDone();
```

`FrameToTensor` resizes, optionally swaps channel order, converts to fp32 and
splits the interleaved HWC buffer into planar CHW directly into the
`ov::Tensor` buffer using `cv::split` on `cv::Mat` headers that alias the
tensor memory — no intermediate allocation.

Note this loop is synchronous: it waits for each result before submitting the
next frame, so it only ever keeps one inference request busy. To exploit a
larger queue, submit several packets before polling, or use
`graph.ObserveOutputStream()` with a callback instead of a poller.

## Changing the graph

`openvino_classification.pbtxt` is a plain text proto — edit `device`,
`plugin_config` and `num_infer_requests` there without rebuilding:

```pbtxt
node {
  calculator: "OpenVINOSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
      model_path: "/home/user/models/resnet50/resnet50.xml"
      device: "CPU"
      plugin_config { key: "PERFORMANCE_HINT" value: "THROUGHPUT" }
      num_infer_requests: 4
    }
  }
}
```

See the [calculator README](../../../calculators/openvino/README.md) for the
full option reference, remote-tensor usage and model sharing between graphs.
