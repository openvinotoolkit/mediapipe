#!/bin/bash
# Copyright 2026 The MediaPipe Authors.
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
# Builds GStreamer from source (following the dlstreamer Ubuntu 24.04
# Dockerfile) and points the MediaPipe @linux_gstreamer repository at it.
# Tested on Ubuntu 24.04.
#
# Full installation:
# $ cd <mediapipe root dir>
# $ bash ./setup_gstreamer.sh
#
# Only update the MediaPipe config for an existing installation:
# $ GSTREAMER_DIR=/path/to/prefix bash ./setup_gstreamer.sh config_only
#
# Then build with:
# $ bazel build --define MEDIAPIPE_ENABLE_GSTREAMER=1 <target>

set -e
if [ "$1" ] && [ "$1" != "config_only" ]
  then
    echo "Unknown input argument. Do you mean \"config_only\"?"
    exit 1
fi

GST_VERSION="${GST_VERSION:-1.28.2}"
GSTREAMER_DIR="${GSTREAMER_DIR:-/opt/gstreamer}"
BUILD_DIR="${BUILD_DIR:-/tmp/build_gstreamer}"

workspace_file="$( cd "$(dirname "$0")" ; pwd -P )"/WORKSPACE

if [ -z "$1" ]
  then
    echo "Installing GStreamer ${GST_VERSION} from source into ${GSTREAMER_DIR}"
    if [[ -x "$(command -v apt)" ]]; then
      echo "skipping apt package installation"
      # sudo apt update
      # sudo apt install -y build-essential git pkg-config flex bison nasm \
      #                 python3-venv libglib2.0-dev libva-dev libdrm-dev \
      #                 libudev-dev libx264-dev libde265-dev libopus-dev \
      #                 libvpx-dev libssl-dev
    elif [[ -x "$(command -v dnf)" ]]; then
      sudo dnf install -y gcc gcc-c++ git pkgconf flex bison nasm python3 \
                       glib2-devel libva-devel libdrm-devel systemd-devel \
                       openssl-devel
    fi

    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # GStreamer 1.28 needs a newer Meson than Ubuntu 24.04 ships.
    python3 -m venv venv
    ./venv/bin/pip install --no-cache-dir meson==1.4.1 ninja==1.11.1.1
    export PATH="${BUILD_DIR}/venv/bin:${PATH}"

    git clone --depth 1 --branch "${GST_VERSION}" \
      https://gitlab.freedesktop.org/gstreamer/gstreamer.git
    cd gstreamer

    meson setup \
      -Dexamples=disabled \
      -Dtests=disabled \
      -Ddoc=disabled \
      -Dintrospection=disabled \
      -Dpython=disabled \
      -Dlibav=disabled \
      -Dlibnice=disabled \
      -Dugly=disabled \
      -Dges=disabled \
      -Drtsp_server=disabled \
      -Ddevtools=disabled \
      -Dgst-examples=disabled \
      -Dorc=disabled \
      -Dgpl=disabled \
      -Dgst-plugins-base:nls=disabled \
      -Dgst-plugins-base:gl=disabled \
      -Dgst-plugins-base:pango=disabled \
      -Dgst-plugins-good:nls=disabled \
      -Dgst-plugins-good:soup=disabled \
      -Dgst-plugins-bad:nls=disabled \
      -Dgst-plugins-bad:doc=disabled \
      -Dgst-plugins-bad:gpl=disabled \
      -Dgst-plugins-bad:va=enabled \
      --buildtype=release \
      --prefix="${GSTREAMER_DIR}" \
      --libdir=lib \
      --libexecdir=bin \
      build
    ninja -C build -j "$(nproc)"
    # sudo resets PATH; meson install needs the venv's ninja.
    sudo env "PATH=${PATH}" meson install -C build --no-rebuild

    cd /
    rm -rf "${BUILD_DIR}"
    echo "GStreamer has been built. Headers: ${GSTREAMER_DIR}/include/gstreamer-1.0, libraries: ${GSTREAMER_DIR}/lib"

    # Plugins dlopen() helper libraries (e.g. libgstcodecs) from the prefix.
    sudo bash -c "echo ${GSTREAMER_DIR}/lib > /etc/ld.so.conf.d/mp_gstreamer.conf"
    sudo ldconfig
fi

echo "Modifying MediaPipe gstreamer config"
linux_gstreamer_config=$(grep -n 'name = "linux_gstreamer"' "$workspace_file" | awk -F ":" '{print $1}')
path_line=$((linux_gstreamer_config + 2))
sed -i "$path_line d" "$workspace_file"
sed -i "$path_line i\    path = \"${GSTREAMER_DIR}\"," "$workspace_file"

echo "Done. At runtime set:"
echo "  export GST_PLUGIN_PATH=${GSTREAMER_DIR}/lib/gstreamer-1.0"
echo "  export GST_PLUGIN_SCANNER=${GSTREAMER_DIR}/bin/gstreamer-1.0/gst-plugin-scanner"
