#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: create_package.sh [--workspace DIR] [--output DIR] [--manifest FILE]
                        [--python-bin EXECUTABLE] [--python-env-prefix DIR]

Manifest entries are tab-separated:
  cpp<TAB>source-path<TAB>package-path
  file<TAB>source-path<TAB>package-path
  python<TAB>source-directory

Paths in the manifest are relative to the workspace. The output directory is
replaced only when it was previously created by this script.
EOF
}
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output=
manifest=
python_bin="${PYTHON_BIN:-python3.12}"
python_env_prefix=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace) workspace="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --manifest) manifest="$2"; shift 2 ;;
        --python-bin) python_bin="$2"; shift 2 ;;
        --python-env-prefix) python_env_prefix="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

workspace="$(cd "$workspace" && pwd)"
[[ -n "$output" ]] || output="$workspace/package"
[[ -n "$manifest" ]] || manifest="$workspace/package.manifest"
manifest="$(realpath -m "$manifest")"
output_parent="$(dirname "$output")"
mkdir -p "$output_parent"
output_parent="$(cd "$output_parent" && pwd)"
output="$output_parent/$(basename "$output")"
if [[ -z "$python_env_prefix" ]]; then
    python_env_prefix="$output/python-env"
elif [[ "$python_env_prefix" != /* ]]; then
    python_env_prefix="$workspace/$python_env_prefix"
fi

if [[ "$output" == / || "$output" == "$workspace" || "$workspace" == "$output/"* ]]; then
    echo "Refusing unsafe package output directory: $output" >&2
    exit 2
fi
if [[ ! -f "$manifest" ]]; then
    echo "Manifest not found: $manifest" >&2
    exit 2
fi
for tool in ldd patchelf "$python_bin"; do
    command -v "$tool" >/dev/null || { echo "Required tool not found: $tool" >&2; exit 1; }
done

marker=.mediapipe-package
if [[ -e "$output" && ! -f "$output/$marker" ]]; then
    echo "Output exists and is not a package created by this script: $output" >&2
    exit 2
fi

stage="$(mktemp -d "$output_parent/.$(basename "$output").tmp.XXXXXX")"
scratch="$(mktemp -d)"
backup=
cleanup() {
    [[ ! -d "$stage" ]] || rm -rf -- "$stage"
    [[ ! -d "$scratch" ]] || rm -rf -- "$scratch"
    if [[ -n "$backup" && -d "$backup" && ! -e "$output" ]]; then
        mv "$backup" "$output"
    fi
}
trap cleanup EXIT

mkdir -p "$stage/libs" "$stage/wheels" "$stage/testdata"
python_source=

copy_library() {
    local library="$1" name destination
    name="$(basename "$library")"
    destination="$stage/libs/$name"
    if [[ -e "$destination" ]]; then
        cmp -s "$library" "$destination" || {
            echo "Library basename collision: $name" >&2
            return 1
        }
    else
        cp -L "$library" "$destination"
    fi
}

copy_runtime_libraries() {
    local binary="$1" ldd_output="$scratch/ldd.log"
    ldd "$binary" > "$ldd_output"
    if grep -Fq 'not found' "$ldd_output"; then
        cat "$ldd_output" >&2
        echo "Unresolved dependency while packaging: $binary" >&2
        return 1
    fi
    while IFS= read -r library; do
        copy_library "$library"
    done < <(awk '$2 == "=>" && $3 ~ /^\// && $3 !~ /^\/(lib|usr\/lib)\// { print $3 }' "$ldd_output")
}

copy_file() {
    local source="$1" destination="$2"
    if [[ "$destination" == /* || "/$destination/" == *"/../"* ]]; then
        echo "Invalid package path: $destination" >&2
        return 1
    fi
    [[ -f "$source" ]] || { echo "Source file not found: $source" >&2; return 1; }
    mkdir -p "$stage/$(dirname "$destination")"
    cp -L "$source" "$stage/$destination"
}

while IFS=$'\t' read -r kind source destination extra || [[ -n "${kind:-}${source:-}${destination:-}${extra:-}" ]]; do
    [[ -z "${kind:-}" || "$kind" == \#* ]] && continue
    [[ -z "${extra:-}" ]] || { echo "Too many manifest fields: $kind $source" >&2; exit 2; }
    case "$kind" in
        cpp)
            [[ -n "${source:-}" && -n "${destination:-}" ]] || { echo "cpp requires source and package path" >&2; exit 2; }
            source="$workspace/$source"
            [[ -x "$source" ]] || { echo "Executable not found: $source" >&2; exit 1; }
            copy_file "$source" "$destination"
            chmod u+w "$stage/$destination"
            chmod +x "$stage/$destination"
            copy_runtime_libraries "$source"
            patchelf --force-rpath --set-rpath '$ORIGIN/libs' "$stage/$destination"
            ;;
        file)
            [[ -n "${source:-}" && -n "${destination:-}" ]] || { echo "file requires source and package path" >&2; exit 2; }
            copy_file "$workspace/$source" "$destination"
            ;;
        python)
            [[ -n "${source:-}" && -z "${destination:-}" ]] || { echo "python requires only a source directory" >&2; exit 2; }
            [[ -z "$python_source" ]] || { echo "Only one Python package is supported per manifest" >&2; exit 2; }
            python_source="$workspace/$source"
            ;;
        *) echo "Unknown manifest entry type: $kind" >&2; exit 2 ;;
    esac
done < "$manifest"

binding=
if [[ -n "$python_source" ]]; then
    binding="$workspace/bazel-bin/${python_source#"$workspace/"}/_openvino_bindings.so"
    [[ -f "$binding" ]] || { echo "Python binding not found: $binding" >&2; exit 1; }
    copy_runtime_libraries "$binding"
fi

for library in "$stage"/libs/*.so*; do
    [[ -e "$library" ]] || continue
    chmod u+w "$library"
    patchelf --force-rpath --set-rpath '$ORIGIN' "$library"
done

if [[ -n "$python_source" ]]; then
    wheel_project="$scratch/wheel-project"
    wheelhouse="$scratch/wheelhouse"
    mkdir -p "$wheel_project/src/mediapipe/python/openvino" "$wheelhouse"
    cp "$python_source"/*.py "$wheel_project/src/mediapipe/python/openvino/"
    cp "$python_source/wheel/setup.py" "$python_source/wheel/pyproject.toml" "$wheel_project/"
    cp -L "$binding" "$wheel_project/src/mediapipe/python/openvino/_openvino_bindings.so"
    chmod u+w "$wheel_project/src/mediapipe/python/openvino/_openvino_bindings.so"
    patchelf --force-rpath --set-rpath '$ORIGIN/../../../../../../../libs' \
        "$wheel_project/src/mediapipe/python/openvino/_openvino_bindings.so"

    OPENVINO_BINDINGS="$wheel_project/src/mediapipe/python/openvino/_openvino_bindings.so" \
        OPENVINO_REQUIREMENTS="$workspace/requirements.txt" \
        "$python_bin" -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir="$wheelhouse" "$wheel_project"
    "$python_bin" -m pip wheel --wheel-dir="$wheelhouse" \
        --only-binary=:all: -r "$workspace/requirements.txt"
    "$python_bin" -m venv "$stage/python-env"
    "$stage/python-env/bin/python" -m pip install \
        --no-index --find-links="$wheelhouse" mediapipe-openvino-demo==0.1.0
    cp "$wheelhouse"/mediapipe_openvino_demo-*.whl "$stage/wheels/"

    for launcher in "$stage/python-env/bin"/*; do
        [[ -f "$launcher" && ! -L "$launcher" ]] || continue
        IFS= read -r shebang < "$launcher" || continue
        old_prefix="#!$stage/python-env/bin/"
        case "$shebang" in
            "$old_prefix"*)
                interpreter="${shebang#"$old_prefix"}"
                rewritten="$launcher.rewrite"
                printf '#!%s/bin/%s\n' "$python_env_prefix" "$interpreter" > "$rewritten"
                tail -n +2 "$launcher" >> "$rewritten"
                chmod --reference="$launcher" "$rewritten"
                mv "$rewritten" "$launcher"
                ;;
        esac
    done
fi

touch "$stage/$marker"
if [[ -e "$output" ]]; then
    backup="$output.backup.$$"
    mv "$output" "$backup"
fi
mv "$stage" "$output"
if [[ -n "$backup" ]]; then
    rm -rf -- "$backup"
    backup=
fi
echo "Package created at $output"