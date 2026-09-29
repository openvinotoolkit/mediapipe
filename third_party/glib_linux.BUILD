# Description:
#   System GLib (apt install libglib2.0-dev), required by GStreamer headers.

licenses(["notice"])  # LGPL

cc_library(
    name = "glib",
    hdrs = glob([
        "include/glib-2.0/**/*.h",
        "lib/x86_64-linux-gnu/glib-2.0/include/*.h",
    ]),
    includes = [
        "include/glib-2.0",
        "lib/x86_64-linux-gnu/glib-2.0/include",
    ],
    linkopts = [
        "-lgobject-2.0",
        "-lglib-2.0",
    ],
    visibility = ["//visibility:public"],
)
