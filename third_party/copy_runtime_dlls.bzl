"""Copies runtime DLLs next to Windows binaries of the calling package."""

def _copy_runtime_dlls_impl(ctx):
    outs = []
    for src in ctx.files.srcs:
        out = ctx.actions.declare_file(src.basename)
        ctx.actions.run(
            executable = "cmd.exe",
            arguments = [
                "/C",
                "copy",
                "/Y",
                src.path.replace("/", "\\"),
                out.path.replace("/", "\\"),
            ],
            inputs = [src],
            outputs = [out],
            mnemonic = "CopyRuntimeDll",
            progress_message = "Copying %s" % src.basename,
            use_default_shell_env = True,
        )
        outs.append(out)
    return [DefaultInfo(
        files = depset(outs),
        runfiles = ctx.runfiles(files = outs),
    )]

copy_runtime_dlls = rule(
    implementation = _copy_runtime_dlls_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".dll"]),
    },
    doc = "Places the given DLLs flat in the package output directory, i.e. next to its cc_binary.",
)
