workspace(name = "ai.applications")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# Protobuf expects an //external:python_headers target
bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)


http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "3.7.2")# ABSL cpp library lts_2023_01_25.

git_repository( # Using commit past 0.9.0 that adds cmake 3.26.2 for model api. Be sure to update to 0.10.0 when available.
    name = "rules_foreign_cc",
    remote = "https://github.com/bazelbuild/rules_foreign_cc.git",
    commit = "1fb8a1e",
 #   strip_prefix = "rules_foreign_cc-0.9.0",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies(cmake_version="3.26.2")

# Start Python setup.py install requirements

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
)

# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.10.4",
  urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

http_archive(
    name = "rules_python",
    sha256 = "29a801171f7ca190c543406f9894abf2d483c206e14d6acbd695623662320097",
    strip_prefix = "rules_python-0.18.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.18.1/rules_python-0.18.1.tar.gz",
)

http_archive(
    name = "stblib",
    strip_prefix = "stb-b42009b3b9d4ca35bc703f5310eedc74f584be58",
    sha256 = "13a99ad430e930907f5611325ec384168a958bf7610e63e60e2fd8e7b7379610",
    urls = ["https://github.com/nothings/stb/archive/b42009b3b9d4ca35bc703f5310eedc74f584be58.tar.gz"],
    build_file = "@mediapipe//third_party:stblib.BUILD",
    patches = [
        "@mediapipe//third_party:stb_image_impl.diff"
    ],
    patch_args = [
        "-p1",
    ],
)

http_archive(
    name = "com_github_glog_glog",
    strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
    sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
    urls = [
        "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
    ],
)

http_archive(
    name = "com_google_audio_tools",
    strip_prefix = "multichannel-audio-tools-1f6b1319f13282eda6ff1317be13de67f4723860",
    urls = ["https://github.com/google/multichannel-audio-tools/archive/1f6b1319f13282eda6ff1317be13de67f4723860.zip"],
    sha256 = "fe346e1aee4f5069c4cbccb88706a9a2b2b4cf98aeb91ec1319be77e07dd7435",
    # TODO: Fix this in AudioTools directly
    patches = ["@mediapipe//third_party:com_google_audio_tools_fixes.diff"],
    patch_args = ["-p1"]
)

http_archive(
    name = "pffft",
    strip_prefix = "jpommier-pffft-7c3b5a7dc510",
    urls = ["https://bitbucket.org/jpommier/pffft/get/7c3b5a7dc510.zip"],
    build_file = "@mediapipe//third_party:pffft.BUILD",
)

http_archive(
    name = "com_google_sentencepiece",
    strip_prefix = "sentencepiece-1.0.0",
    sha256 = "c05901f30a1d0ed64cbcf40eba08e48894e1b0e985777217b7c9036cac631346",
    urls = [
        "https://github.com/google/sentencepiece/archive/1.0.0.zip",
    ],
    patches = [
        "@mediapipe//third_party:com_google_sentencepiece_no_gflag_no_gtest.diff",
    ],
    patch_args = ["-p1"],
    repo_mapping = {"@com_google_glog" : "@com_github_glog_glog"},
)

http_archive(
    name = "zlib",
    build_file = "@mediapipe//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "http://mirror.bazel.build/zlib.net/fossils/zlib-1.2.11.tar.gz",
        "http://zlib.net/fossils/zlib-1.2.11.tar.gz",  # 2017-01-15
    ],
    patches = [
        "@mediapipe//third_party:zlib.diff",
    ],
    patch_args = [
        "-p1",
    ],
)

http_archive(
    name = "org_tensorflow_text",
    sha256 = "f64647276f7288d1b1fe4c89581d51404d0ce4ae97f2bcc4c19bd667549adca8",
    strip_prefix = "text-2.2.0",
    urls = [
        "https://github.com/tensorflow/text/archive/v2.2.0.zip",
    ],
    patches = [
        "@mediapipe//third_party:tensorflow_text_remove_tf_deps.diff",
        "@mediapipe//third_party:tensorflow_text_a0f49e63.diff",
    ],
    patch_args = ["-p1"],
    repo_mapping = {"@com_google_re2": "@com_googlesource_code_re2"},
)

# This is used to select all contents of the archives for CMake-based packages to give CMake access to them.
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

http_archive(
    name = "opencv",
    build_file_content = all_content,
    strip_prefix = "opencv-3.4.10",
    urls = ["https://github.com/opencv/opencv/archive/3.4.10.tar.gz"],
)

# Stop Python setup.py install requirements

git_repository(
    name = "mediapipe",
    remote = "https://github.com/google/mediapipe",
    commit = "d392f8ad98b2d7375e3a57cd3464ecac7efef12a", #  tag: v0.10.3
    patch_args = ["-p1"],
    patches = ["ovms_patch/ovms.patch","ovms_patch/remove_data.patch"]
)

# DEV mediapipe - adjust local repository path for build
#new_local_repository(
#    name = "mediapipe",
#    path = "/fork/",
#    build_file = "/fork/BUILD.bazel",
#)

load("@mediapipe//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

git_repository(
    name = "ovms",
    remote = "https://github.com/openvinotoolkit/model_server",
    commit = "7f372bc9b0a94cf546ef5f1a43e4a9bf768d6f85", # Fix building without MediaPipe (#2129)
)

# TensorFlow repo should always go after the other external dependencies.
# TF on 2023-06-13.
#_TENSORFLOW_GIT_COMMIT = "491681a5620e41bf079a582ac39c585cc86878b9"
_TENSORFLOW_GIT_COMMIT = "491681a5620e41bf079a582ac39c585cc86878b9"
# curl -L https://github.com/tensorflow/tensorflow/archive/<TENSORFLOW_GIT_COMMIT>.tar.gz | shasum -a 256
_TENSORFLOW_SHA256 = "9f76389af7a2835e68413322c1eaabfadc912f02a76d71dc16be507f9ca3d3ac"
http_archive(
    name = "org_tensorflow",
    urls = [
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    patches = [
        "@mediapipe//third_party:org_tensorflow_compatibility_fixes.diff",
        # Diff is generated with a script, don't update it manually.
        "@mediapipe//third_party:org_tensorflow_custom_ops.diff",
        # Logging.h patch
        "@ovms//external:tf.patch",
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    sha256 = _TENSORFLOW_SHA256,
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

########################################################### Python support start

load("@ovms//third_party/python:python_repo.bzl", "python_repository")
python_repository(name = "_python3-linux")

new_local_repository(
    name = "python3_linux",
    path = "/usr",
    build_file = "@_python3-linux//:BUILD"
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    requirements_lock = "@ovms//src:bindings/python/tests/requirements.txt",
)

########################################################### Python support end

load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@//third_party/model_api:model_api.bzl", "model_api_repository")
model_api_repository(name="_model-api")
new_git_repository(
    name = "model_api",
    remote = "https:///github.com/openvinotoolkit/model_api/",
    build_file = "@_model-api//:BUILD",
    commit = "03a6cee5d486ee9eabb625e4388e69fe9c50ef20"
)

# Node dependencies
http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "5aae76dced38f784b58d9776e4ab12278bc156a9ed2b1d9fcd3e39921dc88fda",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.7.1/rules_nodejs-5.7.1.tar.gz"],
)

load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")
build_bazel_rules_nodejs_dependencies()

# fetches nodejs, npm, and yarn
load("@build_bazel_rules_nodejs//:index.bzl", "node_repositories", "yarn_install")
node_repositories()
yarn_install(
    name = "npm",
    package_json = "@mediapipe//:package.json",
    yarn_lock = "@mediapipe//:yarn.lock",
)

# Protobuf for Node dependencies
http_archive(
    name = "rules_proto_grpc",
    sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
    strip_prefix = "rules_proto_grpc-4.2.0",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
)

http_archive(
    name = "com_google_protobuf_javascript",
    sha256 = "35bca1729532b0a77280bf28ab5937438e3dcccd6b31a282d9ae84c896b6f6e3",
    strip_prefix = "protobuf-javascript-3.21.2",
    urls = ["https://github.com/protocolbuffers/protobuf-javascript/archive/refs/tags/v3.21.2.tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr/local",
)

# minitrace
new_git_repository(
    name = "minitrace",
    remote = "https://github.com/hrydgard/minitrace.git",
    commit = "020f42b189e8d6ad50e4d8f45d69edee0a6b3f23",
    build_file_content = """
cc_library(
    name = "trace",
    hdrs = ["minitrace.h"],
    srcs = ["minitrace.c"],
    visibility = ["//visibility:public"],
    local_defines = [
    ],
)
""",
)

# Tensorflow serving
git_repository(
    name = "tensorflow_serving",
    remote = "https://github.com/tensorflow/serving.git",
    tag = "2.13.0",
    patch_args = ["-p1"],
    patches = ["@ovms//external:net_http.patch", "@ovms//external:listen.patch"]
    #                             ^^^^^^^^^^^^
    #                       make bind address configurable
    #          ^^^^^^^^^^^^
    #        allow all http methods
)

# AWS S3 SDK
new_local_repository(
    name = "awssdk",
    build_file = "@ovms//third_party/aws:BUILD",
    path = "/awssdk",
)

# Azure Storage SDK
new_local_repository(
    name = "azure",
    build_file = "@ovms//third_party/azure:BUILD",
    path = "/azure/azure-storage-cpp",
)

# Azure Storage SDK dependency - cpprest
new_local_repository(
    name = "cpprest",
    build_file = "@ovms//third_party/cpprest:BUILD",
    path = "/azure/cpprestsdk",
)

# Boost (needed for Azure Storage SDK)

new_local_repository(
    name = "boost",
    path = "/usr/local/lib/",
    build_file = "@ovms//third_party/boost:BUILD"
)

# Google Cloud SDK
http_archive(
    name = "com_github_googleapis_google_cloud_cpp",
    sha256 = "a370bcf2913717c674a7250c4a310250448ffeb751b930be559a6f1887155f3b",
    strip_prefix = "google-cloud-cpp-0.21.0",
    url = "https://github.com/googleapis/google-cloud-cpp/archive/v0.21.0.tar.gz",
    repo_mapping = {"@com_github_curl_curl" : "@curl"}
)

load("@com_github_googleapis_google_cloud_cpp//bazel:google_cloud_cpp_deps.bzl", "google_cloud_cpp_deps")
google_cloud_cpp_deps()

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,  # C++ support is only "Partially implemented", roll our own.
    grpc = True,
)

load("@com_github_googleapis_google_cloud_cpp_common//bazel:google_cloud_cpp_common_deps.bzl", "google_cloud_cpp_common_deps")
google_cloud_cpp_common_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# cxxopts
http_archive(
    name = "com_github_jarro2783_cxxopts",
    url = "https://github.com/jarro2783/cxxopts/archive/v2.2.0.zip",
    sha256 = "f9640c00d9938bedb291a21f9287902a3a8cee38db6910b905f8eba4a6416204",
    strip_prefix = "cxxopts-2.2.0",
    build_file = "@ovms//third_party/cxxopts:BUILD",
)

# RapidJSON
http_archive(
    name = "com_github_tencent_rapidjson",
    url = "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
    sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
    strip_prefix = "rapidjson-1.1.0",
    build_file = "@ovms//third_party/rapidjson:BUILD"
)

# spdlog
http_archive(
    name = "com_github_gabime_spdlog",
    url = "https://github.com/gabime/spdlog/archive/v1.4.0.tar.gz",
    sha256 = "afd18f62d1bc466c60bef088e6b637b0284be88c515cedc59ad4554150af6043",
    strip_prefix = "spdlog-1.4.0",
    build_file = "@ovms//third_party/spdlog:BUILD"
)

# fmtlib
http_archive(
    name = "fmtlib",
    url = "https://github.com/fmtlib/fmt/archive/6.0.0.tar.gz",
    sha256 = "f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78",
    strip_prefix = "fmt-6.0.0",
    build_file = "@ovms//third_party/fmtlib:BUILD"
)

# libevent
http_archive(
    name = "com_github_libevent_libevent",
    url = "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip",
    sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
    strip_prefix = "libevent-release-2.1.8-stable",
    build_file = "@ovms//third_party/libevent:BUILD",
)

# prometheus-cpp
http_archive(
    name = "com_github_jupp0r_prometheus_cpp",
    strip_prefix = "prometheus-cpp-1.0.1",
    urls = ["https://github.com/jupp0r/prometheus-cpp/archive/refs/tags/v1.0.1.zip"],
)
load("@com_github_jupp0r_prometheus_cpp//bazel:repositories.bzl", "prometheus_cpp_repositories")
prometheus_cpp_repositories()

new_local_repository(
    name = "mediapipe_calculators",
    build_file = "@ovms//third_party/mediapipe_calculators:BUILD",
    path = "/opt/ovms/",
)

new_local_repository(
    name = "linux_openvino",
    build_file = "@ovms//third_party/openvino:BUILD",
    path = "/opt/intel/openvino/runtime",
)

git_repository(
    name = "oneTBB",
    branch = "v2021.10.0",
    remote = "https://github.com/oneapi-src/oneTBB/",
    patch_args = ["-p1"],
    patches = ["@ovms//external:mwaitpkg.patch",]
)