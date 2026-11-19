# MediaPipe OpenVINO Extension

A collection of calculators and application examples for MediaPipe with OpenVINO acceleration for AI models.

Benefits:
- Inference execution boost with OpenVINO Runtime
- Support for Intel accelerators including CPU, iGPU, dGPU, NPU
- Support for Linux and Windows
- Accelearated media content investion and processing with GStreamer and OpenCV

## Examples

[Object detection graph with OpenVINO inference in c++ app](./mediapipe/examples/desktop/openvino_classification/README.md)

[Object detection graph with OpenVINO inference in python](./mediapipe/python/openvino/README.md)



## Linux build environment

Install Bazel 7.7.0 (or Bazelisk), GNU Make, a C++20 compiler, Python 3 with
development headers, Git, CMake and OpenCV's development dependencies. On Ubuntu:

```sh
sudo apt-get install build-essential cmake git python3-dev pkg-config ffmpeg \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk-3-dev \
  libjpeg-dev libpng-dev libtiff-dev libtbb-dev libeigen3-dev
bash setup_opencv.sh
make
```

The setup script builds OpenCV 4.14.0 with the video modules used by the example.
Use an existing OpenCV 4 installation instead with
`OPENCV_PREFIX=/path/to/opencv bash setup_opencv.sh config_only`.
The default Linux prefix is `/usr/local`; `/usr` works for distro OpenCV packages.
Set `OPENCV_PREFIX` in the environment or pass
`BAZEL_FLAGS=--repo_env=OPENCV_PREFIX=/path/to/opencv` to Make.

## Windows build environment

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

## Packaging

After building the C++ and Python targets, create a reusable package from the
workspace's `package.manifest`:

```sh
make build python
./create_package.sh --workspace "$PWD" --output "$PWD/dist/package"
```

Extend the tab-separated manifest to add applications and data files. The
packager gathers native-library dependencies and emits the Python wheel and
venv when a `python` entry is present. For Docker, use `make release_image`
followed by `make test_release_image`; the test video is generated outside the
release image and mounted read-only for the smoke test.

















