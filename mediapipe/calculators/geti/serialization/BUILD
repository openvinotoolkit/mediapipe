cc_library(
    name = "serialization",
    hdrs = [
        "result_serialization.h"
    ],
    deps = [
        "//utils:data_structures",
        "//third_party/cpp-base64:cpp-base64"
    ]
)

filegroup(
    name = "test_data",
    srcs = glob(["*.json"]),
)

cc_library(
    name = "calculators",
    hdrs = [
        "serialization_calculators.h",

    ],
    srcs = [
        "serialization_calculators.cc",

    ],
    deps = [
        "//inference:inference_calculators",
        ":serialization"
    ],
    visibility = [
        "//visibility:public",
    ],
    alwayslink=1
)

cc_test(
    name = "serialization_calculators_test",
    srcs = [ "serialization_calculators_test.cc" ],
    data = [
        ":test_data"
    ],
    deps = [
        ":calculators",
        "//inference:test_deps"
    ]
)


