# Description:
#   OpenVINO Runtime from the official Linux binary (archive) distribution.
#
# The archive layout is:
#   runtime/include/openvino/...          - public C++/C headers
#   runtime/lib/intel64/libopenvino.so*   - core runtime + device plugins + frontends
#   runtime/3rdparty/tbb/lib/*            - oneTBB used by the runtime

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# Device plugins and frontends are dlopen()-ed by libopenvino.so from the
# directory it lives in. They are listed as `srcs` (and not as `data`) so that
# Bazel stages them next to libopenvino.so in the solib directory. Versioned
# libraries (libfoo.so.X) are never linked by Bazel, they are only made
# available at runtime, which is exactly what a dlopen()-ed plugin needs.
#
# The GPU plugin is intentionally excluded because it has a hard NEEDED
# dependency on libOpenCL.so.1 which is not part of this archive. Add it back
# (and install the OpenCL ICD loader) if you need the GPU device.
cc_library(
    name = "openvino",
    srcs = glob(
        [
            "runtime/lib/intel64/*.so",
            "runtime/lib/intel64/*.so.*",
            "runtime/3rdparty/tbb/lib/libtbb.so*",
            "runtime/3rdparty/tbb/lib/libtbbbind_2_5.so*",
            "runtime/3rdparty/tbb/lib/libtbbmalloc.so*",
            "runtime/3rdparty/tbb/lib/libhwloc.so*",
        ],
        exclude = [
            "runtime/lib/intel64/libopenvino_intel_gpu_plugin.so",
        ],
    ),
    hdrs = glob([
        "runtime/include/**/*.h",
        "runtime/include/**/*.hpp",
    ]),
    includes = ["runtime/include"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "openvino_gpu_plugin",
    srcs = ["runtime/lib/intel64/libopenvino_intel_gpu_plugin.so"],
    visibility = ["//visibility:public"],
)
