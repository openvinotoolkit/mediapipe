#!/usr/bin/env bash
set -uo pipefail

status=0
make "$@" || status=$?
if [[ -n "${TEST_RESULTS_DIR:-}" ]]; then
  logs="$(bazel ${BAZEL_STARTUP_FLAGS:-} info bazel-testlogs)" || exit 1
  while IFS= read -r -d '' report; do
    relative="${report#"$logs"/}"
    destination="$TEST_RESULTS_DIR/$relative"
    mkdir -p "$(dirname "$destination")"
    cp "$report" "$destination"
  done < <(find -L "$logs" -type f \( -name test.xml -o -name test.log \) -print0)
fi
exit "$status"