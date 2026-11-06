# MediaPipe OpenVINO Extension

Custom OpenVINO calculators and classification examples for MediaPipe, without
vendoring the upstream MediaPipe source tree.

## Contents

- `mediapipe/calculators/openvino`: session creation, preprocessing, inference,
  shared inference-request pools, calculator tests and small OpenVINO IR models.
- `mediapipe/examples/desktop/openvino_classification`: C++ video classification
  application, example graph and an end-to-end test using a generated video.
- `mediapipe/python/openvino`: standalone Python bindings and classification demo.
- `third_party`: OpenVINO/OpenCV Bazel adapters and compatibility patches.
- `Makefile`, `Dockerfile` and `ci/testOnCommit.groovy`: local builds and CI.

`WORKSPACE` downloads Google AI Edge MediaPipe at commit
`8ac5a39c659578c2595a54ef5277608173c217fe`, verified by SHA-256. Its framework,
build definitions, third-party patches and TensorFlow lockfiles come from that
archive. The retained dependency declarations bootstrap upstream's transitive
repositories; Bazel does not evaluate an external repository's WORKSPACE for you.
`MODULE.bazel` supplies the matching rule modules, protobuf/Abseil versions and
Python 3.11 tooling. Keep both WORKSPACE and Bzlmod enabled with Bazel 7.7.0.

OpenVINO Runtime 2026.4.0 is fetched by Bazel for Linux x86-64 or Windows x64.
The Linux distribution requires Ubuntu 24.04-compatible runtime libraries.
CPU inference is the default; the Linux GPU plugin is intentionally not linked.

## Linux

Install Bazel 7.7.0 (or Bazelisk), GNU Make, a C++20 compiler, Python 3 with
development headers, Git, CMake and OpenCV's development dependencies. On Ubuntu:

```sh
sudo apt-get install build-essential cmake git python3-dev pkg-config ffmpeg \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk-3-dev \
  libjpeg-dev libpng-dev libtiff-dev libtbb-dev libeigen3-dev
bash setup_opencv.sh
make
```

The setup script builds OpenCV 4.13.0 with the video modules used by the example.
Use an existing OpenCV 4 installation instead with
`OPENCV_PREFIX=/path/to/opencv bash setup_opencv.sh config_only`.
The default Linux prefix is `/usr/local`; `/usr` works for distro OpenCV packages.
Set `OPENCV_PREFIX` in the environment or pass
`BAZEL_FLAGS=--repo_env=OPENCV_PREFIX=/path/to/opencv` to Make.

## Windows

Use Windows x64 with Visual Studio 2022's Desktop Development with C++ workload,
CMake and Git for Windows. Run from `cmd.exe`:

```bat
call prepare_build_dependencies.bat
C:\opt\msys64\usr\bin\make.exe build tests example-tests
```

The setup script installs Bazel, MSYS2, GNU Make and Python under `C:\opt`, and
builds OpenCV 4.13.0 with `opencv_world` and FFmpeg support. It exports the build
environment to the calling command prompt. Bazel downloads OpenVINO itself.
`OPENCV_PREFIX` can override the default `C:/opt/opencv` location.
The Makefile defaults to `--output_user_root=C:/b` on Windows to limit path length.
Native Bazel commands work too:

```bat
bazel --output_user_root=C:/b build //mediapipe/examples/desktop/openvino_classification:openvino_classification
bazel --output_user_root=C:/b test //mediapipe/calculators/openvino:openvino_session_calculator_test //mediapipe/calculators/openvino:openvino_calculators_concurrency_test
```

The Windows patches retain the MSVC protobuf, Java and Swift toolchain fixes.
MediaPipe framework fixes are applied to the downloaded archive. C++20 and the
conforming MSVC preprocessor are enabled for both local and external sources.
OpenVINO/OpenCV runtime DLLs are staged next to the binaries; test assets use
Bazel runfiles lookup, including Windows manifests.

## Build Targets

| Command | Result |
| --- | --- |
| `make build` | Calculator libraries and C++ classification application |
| `make tests` | Calculator preprocessing, reshaping and concurrency tests |
| `make example-tests` | C++ video/inference smoke test with a local IR model |
| `make test` | All included C++ tests |
| `make python` | Python extension and classification demo |
| `make clean` | Clean Bazel build outputs |

`make` builds the C++ components and runs all C++ tests. Override `BAZEL`,
`BAZEL_STARTUP_FLAGS`, `BAZEL_FLAGS` and `JOBS` as needed. The Python bindings
remain a separate target; do not import them alongside a separately installed
MediaPipe extension, which would duplicate pybind type registrations.

## Run Classification

Place an OpenVINO classification IR at `models/resnet50/resnet50.xml` with its
matching `.bin`, or edit `model_path` in the example graph. Match input size,
layout and normalization to the model:

```sh
bazel run //mediapipe/examples/desktop/openvino_classification:openvino_classification -- \
  --calculator_graph_config_file="$PWD/mediapipe/examples/desktop/openvino_classification/openvino_classification.pbtxt" \
  --input_video_path=/path/to/video.mp4 --input_width=224 --input_height=224
```

Use absolute paths with `bazel run`. On Windows, use the `.exe` under
`bazel-bin\mediapipe\examples\desktop\openvino_classification` and the same flags.
The package READMEs document calculator options and Python usage.

## CI

The Jenkins pipeline adapts the build-image, demos-image, unit-test and C++-demo
stages from the OpenVINO toolkit MediaPipe pipeline. Configure the Jenkins job's
script path as `ci/testOnCommit.groovy`. It requires a Docker-capable Linux agent
(`ovmscheck` by default) and a Windows agent (`windows` by default); both labels
are parameters. The Windows agent must have Visual Studio, CMake and Git, plus
permission to install dependencies in `C:\opt`.

```sh
make docker_build ovms_demos_image
make tests_in_docker run_demos_in_docker
```

Both native and container tests produce Bazel JUnit XML. Docker targets export
results to `test-results`, and Jenkins collects XML and logs on both platforms.
The Docker image uses Ubuntu 24.04 and the same OpenCV setup script as local builds.

## Scope And License

Only custom extension sources and required build integration belong in this
repository. Update the pinned MediaPipe archive and compatibility patches together.
The pre-existing executable-bit change to `build_desktop_examples.sh` and local
untracked SDK/install files were preserved during extraction; the old desktop
script is not part of the extension build and still refers to upstream examples.

Apache License 2.0; see `LICENSE`. Downloaded dependencies retain their own licenses.