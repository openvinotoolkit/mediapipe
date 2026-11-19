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

## Build and run on Linux

Use Ubuntu 24.04 x86-64 with Bazel 7.7.0 (or Bazelisk). Run the following
from the repository root in Bash:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl python3-dev pkg-config ffmpeg \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk-3-dev \
  libjpeg-dev libpng-dev libtiff-dev libtbb-dev libeigen3-dev
bash setup_opencv.sh
sudo bash install_drivers.sh
```

The OpenCV setup script installs to `/usr/local` by default; use a writable
`OPENCV_PREFIX` or the installation permissions described in the
[Linux build instructions](../../../../README.md#linux-build-environment).
Bazel downloads OpenVINO Runtime. GPU execution additionally requires a supported
Intel GPU, a compatible host kernel driver, and access to `/dev/dri/renderD*`
(usually through membership in the `render` group; log in again after changing
group membership). Installing userspace drivers alone does not provide a GPU.

Download the model and generate the video as above, then create a local graph
with an absolute model path. This leaves the checked-in graph unchanged:

```bash
mkdir -p test-results
sed "s|model_path:.*|model_path: \"$HOME/models/resnet50/resnet50.xml\"|" \
  mediapipe/examples/desktop/openvino_classification/openvino_classification.pbtxt \
  > test-results/resnet50-cpu.pbtxt

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  //mediapipe/examples/desktop/openvino_classification:openvino_classification

bazel-bin/mediapipe/examples/desktop/openvino_classification/openvino_classification \
  --calculator_graph_config_file="$PWD/test-results/resnet50-cpu.pbtxt" \
  --input_video_path="$HOME/models/test_video.mp4" \
  --max_frames=5
```

`MEDIAPIPE_DISABLE_GPU=1` disables MediaPipe's GPU subsystem, not OpenVINO GPU
inference. To run on the Intel GPU, change `device: "CPU"` to `device: "GPU"`
in the local graph and use the same executable.

```
OpenVINOSessionCalculator compiled .../resnet50.xml on CPU with 4 inference requests
frame 0 output[0] shape=[1,1000] type=f32
  values: -2.39221 1.0669 -3.18584 ...
  top5: 419=6.91352 906=6.62288 916=6.28421 549=6.15924 902=5.99802
```

### Benchmark ResNet50 on CPU and GPU

The [benchmark script](benchmark.sh) runs one application with one source and
then one application with 20 producer threads and 20 MediaPipe graphs on each
device. All graphs share one compiled model and one inference-request pool via
the session registry, as in the
[concurrency test](../../../calculators/openvino/openvino_calculators_concurrency_test.cc).
Each thread reads the same input file through its own decoder and sends packets
to its own graph; this simulates independent sources, not 20 distinct video
contents or network streams.

Generate a longer input first (the short video above has only 20 frames):

```bash
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i testsrc=size=640x480:rate=30:duration=60 \
  -pix_fmt yuv420p -y "$HOME/models/benchmark_video.mp4"
```

After building the executable, run this from the repository root:

```bash
bash mediapipe/examples/desktop/openvino_classification/benchmark.sh
```

The defaults are 1,000 frames per source, devices `CPU GPU`, and client counts
`1 20`. Override inputs or select a smaller matrix with environment variables:

```bash
MODEL="$HOME/models/resnet50/resnet50.xml" \
VIDEO="$HOME/models/benchmark_video.mp4" \
FRAMES=1000 DEVICES="CPU GPU" CLIENT_COUNTS="1 20" \
  bash mediapipe/examples/desktop/openvino_classification/benchmark.sh
```

Each run saves `results.csv`, generated graphs, and one `run.log` per case in a new
`test-results/resnet50-benchmark.*` directory. `OUTPUT_DIR` and `DEMO` can override
the results directory and executable. A failed application, incomplete per-source
frame counts, or a compilation count other than one marks the case `FAILED`;
the other cases still run, and the script exits nonzero if any case fails.

These are **video-processing throughput** measurements, including decoder
startup, video decoding, preprocessing, packet transfers, inference, and graph
draining. Model compilation and graph initialization finish before the timer
starts; tensor printing is disabled with `--print_tensors=false`. There is no
warm-up phase, so the first inference is included. These are not inference-only
latency measurements. Inputs run as fast as possible, without video frame-rate
pacing. One source uses the `LATENCY` hint and one request; 20 sources use
`THROUGHPUT` and a shared pool of 20 requests. Each producer waits for its result
before submitting its next frame. The producer count does not include MediaPipe,
decoder, or OpenVINO internal worker threads. Run the four cases sequentially,
repeat for comparisons, and use OpenVINO's `benchmark_app` for inference-only
measurements. Earlier results from the multi-process script are not comparable.

## Build and run on Windows

Prerequisite: Visual Studio 2022 with the "Desktop development with C++"
workload (including "C++ CMake tools for Windows") and Git for Windows.

From `cmd.exe` in the repository root, install MSYS2, Python, Bazel and OpenCV
into `C:\opt` and set up the build environment for the current session
(rerun it in every new `cmd.exe`; already installed parts are skipped):

```
call prepare_build_dependencies.bat
```

OpenVINO Runtime is downloaded by Bazel. A short `--output_user_root` avoids
Windows path length limits:

```
bazel --output_user_root=C:/b build -c opt --define MEDIAPIPE_DISABLE_GPU=1 ^
  //mediapipe/examples/desktop/openvino_classification:openvino_classification

bazel-bin\mediapipe\examples\desktop\openvino_classification\openvino_classification.exe ^
  --calculator_graph_config_file=mediapipe/examples/desktop/openvino_classification/openvino_classification.pbtxt ^
  --input_video_path=C:\models\test_video.mp4 ^
  --max_frames=5
```

The OpenVINO, TBB and OpenCV DLLs (including the OpenVINO device plugins and
the OpenCV FFmpeg plugin) are copied next to the `.exe`, so no `setupvars.bat`
or `PATH` changes are needed to run it. Use a Windows path for `model_path` in
the `.pbtxt`, e.g. `model_path: "C:/models/resnet50/resnet50.xml"`.

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
| `--num_sources` | 1 | Producer threads and graphs in one process, sharing a model session. Each independently reads `--input_video_path`. |
| `--print_tensors` | true | Print frame tensors; set false for benchmarks. Source counts and timing are still printed. |

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

Each source's loop is synchronous: it waits for each result before submitting
the next frame, keeping at most one inference request busy per source. With
`--num_sources=20`, all 20 graphs are started before producer threads run, so
their sessions overlap and share one compiled model/request pool. Set
`num_infer_requests` in the graph to control that shared pool's capacity.

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
