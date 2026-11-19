# Description:
#   OpenVINO Runtime from the official Windows binary (zip) distribution.
#
# The archive layout is:
#   runtime/include/openvino/...                 - public C++/C headers
#   runtime/lib/intel64/Release/openvino.lib     - import library
#   runtime/bin/intel64/Release/*.dll            - core runtime + plugins + frontends
#   runtime/3rdparty/tbb/bin/*.dll               - oneTBB used by the runtime

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# system_provided: the DLLs are staged next to the binary via :runtime_dlls,
# because plugins are loaded from the directory of openvino.dll.
cc_import(
    name = "openvino_import_lib",
    interface_library = "runtime/lib/intel64/Release/openvino.lib",
    system_provided = 1,
)

cc_library(
    name = "openvino",
    hdrs = glob([
        "runtime/include/**/*.h",
        "runtime/include/**/*.hpp",
    ]),
    includes = ["runtime/include"],
    visibility = ["//visibility:public"],
    deps = [":openvino_import_lib"],
)

filegroup(
    name = "runtime_dlls",
    srcs = glob(
        [
            "runtime/bin/intel64/Release/*.dll",
            "runtime/3rdparty/tbb/bin/*.dll",
        ],
        exclude = [
            "runtime/3rdparty/tbb/bin/*_debug.dll",
            "runtime/3rdparty/tbb/bin/tbbmalloc_proxy.dll",
        ],
    ),
    visibility = ["//visibility:public"],
)
