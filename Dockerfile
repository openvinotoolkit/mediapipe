FROM ubuntu:24.04 AS dependencies

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl ffmpeg git libavcodec-dev \
    libavformat-dev libavutil-dev libeigen3-dev libgtk-3-dev libjpeg-dev \
    libpng-dev libswscale-dev libtbb-dev libtiff-dev patch pkg-config \
    python3-dev unzip zip && rm -rf /var/lib/apt/lists/*

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