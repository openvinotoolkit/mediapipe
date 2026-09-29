# Description:
#   GStreamer installed by setup_gstreamer.sh (default prefix /opt/gstreamer).

licenses(["notice"])  # LGPL

cc_library(
    name = "gstreamer",
    srcs = glob([
        "lib/libgstreamer-1.0.so*",
        "lib/libgstbase-1.0.so*",
        "lib/libgstapp-1.0.so*",
        "lib/libgstvideo-1.0.so*",
        "lib/libgstallocators-1.0.so*",
        "lib/libgstpbutils-1.0.so*",
        "lib/libgstva-1.0.so*",
    ]),
    hdrs = glob(["include/gstreamer-1.0/**/*.h"]),
    includes = ["include/gstreamer-1.0"],
    visibility = ["//visibility:public"],
    deps = ["@linux_glib//:glib"],
)

filegroup(
    name = "plugins",
    srcs = glob(["lib/gstreamer-1.0/*.so"]),
    visibility = ["//visibility:public"],
)
