# Description:
#   OpenCV libraries for video/image processing on Windows.
#
# Expects an opencv_world build, e.g. the one produced by
# prepare_build_dependencies.bat (C:\opt\opencv). Both the flat layout
# (lib/, bin/) and the default MSVC layout (x64/vc*/lib, x64/vc*/bin) are
# supported.

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
)

_OPT_LIBS = glob(
    [
        "lib/opencv_world*.lib",
        "x64/*/lib/opencv_world*.lib",
    ],
    exclude = [
        "lib/opencv_world*d.lib",
        "x64/*/lib/opencv_world*d.lib",
    ],
    allow_empty = True,
)

_DBG_LIBS = glob(
    [
        "lib/opencv_world*d.lib",
        "x64/*/lib/opencv_world*d.lib",
    ],
    allow_empty = True,
)

_OPT_LIB = _OPT_LIBS[0] if _OPT_LIBS else fail(
    "opencv_world*.lib not found in the windows_opencv repository. " +
    "Run prepare_build_dependencies.bat or fix the path in WORKSPACE.",
)

# system_provided: the DLLs are staged next to the binary via :runtime_dlls.
cc_import(
    name = "opencv_import_lib",
    interface_library = select({
        ":dbg_build": _DBG_LIBS[0] if _DBG_LIBS else _OPT_LIB,
        "//conditions:default": _OPT_LIB,
    }),
    system_provided = 1,
)

cc_library(
    name = "opencv",
    hdrs = glob(["include/opencv2/**/*.h*"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
    deps = [":opencv_import_lib"],
)

# Includes the dynamically loaded videoio FFmpeg plugin.
filegroup(
    name = "runtime_dlls",
    srcs = select({
        ":dbg_build": glob(
            [
                "bin/*.dll",
                "x64/*/bin/*.dll",
            ],
            allow_empty = True,
        ),
        "//conditions:default": glob(
            [
                "bin/*.dll",
                "x64/*/bin/*.dll",
            ],
            exclude = [
                "bin/opencv_world*d.dll",
                "x64/*/bin/opencv_world*d.dll",
            ],
            allow_empty = True,
        ),
    }),
    visibility = ["//visibility:public"],
)
