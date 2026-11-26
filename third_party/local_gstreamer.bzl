"""GStreamer development headers from /opt/gstreamer (source build) or /usr."""

def _local_gstreamer_impl(ctx):
    prefix = ctx.os.environ.get("GSTREAMER_PREFIX", "")
    if not prefix:
        prefix = "/usr"
        for candidate in ctx.attr.default_paths:
            if ctx.path(candidate + "/include/gstreamer-1.0").exists:
                prefix = candidate
                break
    for child in ["include", "lib"]:
        path = ctx.path(prefix + "/" + child)
        if path.exists:
            ctx.symlink(path, child)
    ctx.file("WORKSPACE", "")
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

local_gstreamer = repository_rule(
    implementation = _local_gstreamer_impl,
    attrs = {
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "default_paths": attr.string_list(mandatory = True),
    },
    environ = ["GSTREAMER_PREFIX"],
    local = True,
)
