#!/usr/bin/env bash
set -euo pipefail

if [[ "$(dpkg --print-architecture)" != amd64 ]]; then
    echo "Intel GPU driver installer supports amd64 only" >&2
    exit 2
fi
if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run install_drivers.sh as root" >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl libnuma1 ocl-icd-libopencl1

gpu_deps="$(mktemp -d)"
cleanup() {
    rm -rf -- "$gpu_deps" /var/lib/apt/lists/*
}
trap cleanup EXIT

compute_runtime_url=https://github.com/intel/compute-runtime/releases/download/26.31.39395.13
igc_url=https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13
level_zero_url=https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb
level_zero_sha256=d4001dc474d3cb74ded50a449b9f646cfdfbb31e69e70b8af38f2d18273dfce1

for url in \
    "$igc_url/intel-igc-core-2_2.40.13+22418_amd64.deb" \
    "$igc_url/intel-igc-opencl-2_2.40.13+22418_amd64.deb" \
    "$compute_runtime_url/intel-ocloc_26.31.39395.13-0_amd64.deb" \
    "$compute_runtime_url/intel-opencl-icd_26.31.39395.13-0_amd64.deb" \
    "$compute_runtime_url/libigdgmm12_22.10.0_amd64.deb" \
    "$compute_runtime_url/libze-intel-gpu1_26.31.39395.13-0_amd64.deb" \
    "$level_zero_url"; do
    curl --fail --location --retry 3 "$url" --output "$gpu_deps/${url##*/}"
done

echo "$level_zero_sha256  $gpu_deps/${level_zero_url##*/}" | sha256sum --check -
apt-get install -y --no-install-recommends "$gpu_deps"/*.deb
apt-get clean
