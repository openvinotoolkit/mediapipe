# Description:
#   GStreamer and GLib development headers, either from the source build in
#   /opt/gstreamer (setup_gstreamer.sh copies GLib headers there) or from the
#   Ubuntu packages libgstreamer1.0-dev and libgstreamer-plugins-base1.0-dev.
#
# Headers only: the Intel MPI loader dlopen()s the GStreamer libraries at
# runtime, so binaries start without GStreamer installed.

licenses(["restricted"])  # LGPL-2.1+

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gstreamer",
    hdrs = glob(
        [
            "include/gstreamer-1.0/**/*.h",
            "include/glib-2.0/**/*.h",
            "lib/x86_64-linux-gnu/glib-2.0/include/*.h",
        ],
        allow_empty = True,
    ),
    includes = [
        "include/gstreamer-1.0",
        "include/glib-2.0",
        "lib/x86_64-linux-gnu/glib-2.0/include",
    ],
)
