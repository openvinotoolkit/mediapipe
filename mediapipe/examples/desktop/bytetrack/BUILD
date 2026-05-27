licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])


cc_binary(
    name = "bytetrack_final_cpu",
    data = [
        "//mediapipe/models:yolox_nano_float32",
        "//mediapipe/models:coco_labels",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/bytetrack:bytetrack_cpu_calculators",
    ],   
)

cc_binary(
    name = "bytetrack_final_ovms",
    data = [
        "//mediapipe/models:coco_labels",
    ],
    deps = [
        "@ovms//src:ovms_lib",
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/bytetrack:bytetrack_ovms_calculators",
    ],   
)
