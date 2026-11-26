# GStreamer for //mediapipe/calculators/gstreamer (--config=gst), dlopen()-ed
# at runtime. GSTREAMER_SOURCE=build (default) builds 1.26 from source into
# /opt/gstreamer; GSTREAMER_SOURCE=apt installs Ubuntu's 1.24.2 instead.
# WARNING: Ubuntu's 1.24.2 vapostproc crashes (GLib "failed to allocate") when
# scaling VA frames into system memory, e.g. GstVideoSourceCalculator
# FRAME_FULL host frames at a size other than the source. GPU surface paths
# work with both versions.
FROM ubuntu:24.04 AS gstreamer
ARG GSTREAMER_SOURCE=build
ARG GST_VERSION=1.26.2
ARG JOBS=16
COPY setup_gstreamer.sh /tmp/setup_gstreamer.sh
RUN if [ "${GSTREAMER_SOURCE}" = build ]; then \
        GST_VERSION="${GST_VERSION}" JOBS="${JOBS}" /tmp/setup_gstreamer.sh; \
    else \
        mkdir -p /opt/gstreamer; \
    fi

FROM ubuntu:24.04 AS dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl ffmpeg git libavcodec-dev \
    libavformat-dev libavutil-dev libeigen3-dev libgtk-3-dev libjpeg-dev \
    libpng-dev libswscale-dev libtbb-dev libtiff-dev patchelf patch \
    pkg-config unzip zip && rm -rf /var/lib/apt/lists/*

COPY install_drivers.sh /usr/local/bin/install_drivers.sh
RUN /usr/local/bin/install_drivers.sh

# VA media driver. Ubuntu's own iHD crashes with the gmmlib from install_drivers.sh.
# MEDIA_DRIVER_SOURCE=intel-gpu-repo: Intel's GPU repository (iHD 24.3).
# MEDIA_DRIVER_SOURCE=kobuk-ppa: ppa:kobuk-team/intel-graphics, newer iHD; needed for Arc B-series.
# MEDIA_DRIVER_VERSION pins intel-media-va-driver-non-free (apt version); empty = latest.
ARG MEDIA_DRIVER_SOURCE=intel-gpu-repo
ARG MEDIA_DRIVER_VERSION=
RUN apt-get update && apt-get install -y --no-install-recommends gnupg && \
    case "${MEDIA_DRIVER_SOURCE}" in \
    intel-gpu-repo) \
        curl --fail --location --retry 3 https://repositories.intel.com/gpu/intel-graphics.key | \
        gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" \
        > /etc/apt/sources.list.d/intel-gpu-noble.list ;; \
    kobuk-ppa) \
        fpr=0C0E6AF955CE463C03FC51574D098D70AFBE5E1F && \
        curl --fail --location --retry 3 "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${fpr}" | \
        gpg --dearmor --output /usr/share/keyrings/kobuk-intel-graphics.gpg && \
        gpg --show-keys --with-colons /usr/share/keyrings/kobuk-intel-graphics.gpg | grep -qx "fpr:::::::::${fpr}:" && \
        echo "deb [signed-by=/usr/share/keyrings/kobuk-intel-graphics.gpg] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
        > /etc/apt/sources.list.d/kobuk-intel-graphics.list ;; \
    *) echo "MEDIA_DRIVER_SOURCE must be intel-gpu-repo or kobuk-ppa" >&2; exit 1 ;; \
    esac && \
    apt-get update && apt-get install -y --no-install-recommends \
    "intel-media-va-driver-non-free${MEDIA_DRIVER_VERSION:+=${MEDIA_DRIVER_VERSION}}" \
    libdrm2 libglib2.0-0t64 libva-dev libva-drm2 libva2 && \
    rm -rf /var/lib/apt/lists/*

ARG GSTREAMER_SOURCE=build
COPY --from=gstreamer /opt/gstreamer /opt/gstreamer
RUN apt-get update && \
    if [ "${GSTREAMER_SOURCE}" = build ]; then \
        apt-get install -y --no-install-recommends libopenh264-7 && \
        echo /opt/gstreamer/lib > /etc/ld.so.conf.d/gstreamer.conf && ldconfig; \
    else \
        apt-get install -y --no-install-recommends gstreamer1.0-libav \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good gstreamer1.0-rtsp \
        libgstreamer-plugins-bad1.0-0 libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev; \
    fi && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/gstreamer/bin:${PATH}"
ENV GST_PLUGIN_PATH=/opt/gstreamer/lib/gstreamer-1.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-setuptools python3-wheel python3.12 \
    python3.12-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY .bazelversion setup_opencv.sh ./
RUN BAZEL_VERSION="$(tr -d '\r\n' < .bazelversion)" && \
    curl --fail --location --retry 3 \
    "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64" \
    -o /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
ARG JOBS=16
RUN JOBS="${JOBS}" bash setup_opencv.sh && ldconfig

FROM dependencies AS builder
COPY . .
ARG JOBS=16
RUN make build python JOBS="${JOBS}"
RUN ./create_package.sh --workspace /workspace --output /workspace/runtime-root \
    --python-env-prefix /mediapipe/python-env

FROM builder AS demos
CMD ["make", "example-tests"]

FROM ubuntu:24.04 AS release
ENV DEBIAN_FRONTEND=noninteractive

COPY install_drivers.sh /usr/local/bin/install_drivers.sh
RUN /usr/local/bin/install_drivers.sh && \
    apt-get purge -y curl && apt-get autoremove -y && \
    rm -f /usr/local/bin/install_drivers.sh && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends python3.12 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /mediapipe
COPY --from=builder /workspace/runtime-root/ /mediapipe/
RUN set -eu; \
    ldd /mediapipe/openvino_classification > /tmp/cpp-ldd.log; \
    ! grep -Fq 'not found' /tmp/cpp-ldd.log; \
    binding="$(find /mediapipe/python-env/lib/python3.12/site-packages/mediapipe/python/openvino -maxdepth 1 -name '_openvino_bindings*.so' -print -quit)"; \
    test -n "${binding}" && test -f "${binding}"; \
    ldd "${binding}" > /tmp/python-ldd.log; \
    cat /tmp/python-ldd.log; \
    ! grep -Fq 'not found' /tmp/python-ldd.log; \
    /mediapipe/python-env/bin/python3.12 -c 'import cv2, numpy; from mediapipe.python import openvino'; \
    rm -f /tmp/cpp-ldd.log /tmp/python-ldd.log
ENTRYPOINT ["/mediapipe/python-env/bin/openvino-classification-demo"]