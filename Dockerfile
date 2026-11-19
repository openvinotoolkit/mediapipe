FROM ubuntu:24.04 AS dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl ffmpeg git libavcodec-dev \
    libavformat-dev libavutil-dev libeigen3-dev libgtk-3-dev libjpeg-dev \
    libpng-dev libswscale-dev libtbb-dev libtiff-dev patchelf patch \
    pkg-config unzip zip && rm -rf /var/lib/apt/lists/*

COPY install_drivers.sh /usr/local/bin/install_drivers.sh
RUN /usr/local/bin/install_drivers.sh

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