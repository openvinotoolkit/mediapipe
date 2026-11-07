#!/bin/bash
# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#
# Build OpenCV 4 with the desktop video modules, or configure an existing install.

set -euo pipefail

case "${1:-}" in
  ""|config_only) ;;
  *) echo "Usage: bash setup_opencv.sh [config_only]" >&2; exit 1 ;;
esac

repo_dir="$(cd "$(dirname "$0")" && pwd -P)"
prefix="${OPENCV_PREFIX:-/usr/local}"
version="${OPENCV_VERSION:-4.14.0}"

if [[ "${1:-}" != config_only ]]; then
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/opencv.XXXXXXXX")"
  trap 'rm -rf "$build_dir"' EXIT
  git clone --depth 1 --branch "$version" https://github.com/opencv/opencv.git "$build_dir/opencv"
  git clone --depth 1 --branch "$version" https://github.com/opencv/opencv_contrib.git "$build_dir/opencv_contrib"
  cmake -S "$build_dir/opencv" -B "$build_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DOPENCV_EXTRA_MODULES_PATH="$build_dir/opencv_contrib/modules" \
    -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,video,videoio,optflow \
    -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
    -DWITH_FFMPEG=ON -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_TIFF=ON \
    -DCV_ENABLE_INTRINSICS=ON -DWITH_EIGEN=ON -DWITH_PTHREADS=ON
  cmake --build "$build_dir/build" --parallel "${JOBS:-16}"
  if [[ -w "$prefix" ]] || [[ ! -e "$prefix" && -w "$(dirname "$prefix")" ]]; then
    cmake --install "$build_dir/build"
  else
    sudo cmake --install "$build_dir/build"
  fi
fi

if [[ ! -f "$prefix/include/opencv4/opencv2/core.hpp" ]]; then
  echo "OpenCV 4 headers not found under $prefix; set OPENCV_PREFIX to its install prefix" >&2
  exit 1
fi
prefix="$(cd "$prefix" && pwd -P)"
if [[ "$prefix" == *'"'* || "$prefix" == *$'\n'* ]]; then
  echo "OpenCV prefix cannot contain quotes or newlines" >&2
  exit 1
fi
printf 'common --repo_env=OPENCV_PREFIX="%s"\n' "$prefix" > "$repo_dir/.configure.bazelrc"
echo "OpenCV $prefix configured in .configure.bazelrc; tracked Bazel files are unchanged."
