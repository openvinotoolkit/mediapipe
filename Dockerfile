FROM ubuntu:24.04 AS dependencies

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl ffmpeg git libavcodec-dev \
    libavformat-dev libavutil-dev libeigen3-dev libgtk-3-dev libjpeg-dev \
    libnuma1 libpng-dev libswscale-dev libtbb-dev libtiff-dev \
    ocl-icd-libopencl1 patch pkg-config \
    python3-dev unzip zip && rm -rf /var/lib/apt/lists/*

# OpenVINO loads its GPU/NPU plugins even when CPU is selected. Install their
# ICD/Level Zero loaders; physical acceleration still requires host devices and
# the corresponding vendor drivers.
ARG LEVEL_ZERO_DEB_URL=https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb
ARG LEVEL_ZERO_DEB_SHA256=d4001dc474d3cb74ded50a449b9f646cfdfbb31e69e70b8af38f2d18273dfce1
RUN curl --fail --location --retry 3 "${LEVEL_ZERO_DEB_URL}" -o /tmp/libze1.deb && \
    echo "${LEVEL_ZERO_DEB_SHA256}  /tmp/libze1.deb" | sha256sum --check - && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/libze1.deb && \
    rm -f /tmp/libze1.deb && rm -rf /var/lib/apt/lists/*

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
RUN make build JOBS="${JOBS}"

FROM builder AS demos
CMD ["make", "example-tests"]