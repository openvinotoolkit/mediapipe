"""Installed OpenCV with a configurable prefix on Linux and Windows."""

def _local_opencv_impl(ctx):
    prefix = ctx.os.environ.get("OPENCV_PREFIX", ctx.attr.default_path)
    root = ctx.path(prefix)
    if not root.exists:
        fail("OpenCV prefix %s does not exist; run the setup script or set OPENCV_PREFIX" % prefix)
    for child in root.readdir():
        if child.basename not in ["BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel"]:
            ctx.symlink(child, child.basename)
    ctx.file("WORKSPACE", "")
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

local_opencv = repository_rule(
    implementation = _local_opencv_impl,
    attrs = {
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "default_path": attr.string(mandatory = True),
    },
    environ = ["OPENCV_PREFIX"],
    local = True,
)