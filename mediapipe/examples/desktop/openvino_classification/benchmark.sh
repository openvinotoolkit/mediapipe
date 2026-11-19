#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
    printf '%s\n' \
        'Usage: bash benchmark.sh' \
        'Environment: MODEL, VIDEO, DEMO, OUTPUT_DIR, FRAMES (1000),' \
        '             DEVICES ("CPU GPU"), CLIENT_COUNTS ("1 20").' \
        'Measures video throughput using one process and a shared model session.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Unexpected argument; use --help.' >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_dir/../../../.." && pwd)"
MODEL="${MODEL:-$HOME/models/resnet50/resnet50.xml}"
VIDEO="${VIDEO:-$HOME/models/benchmark_video.mp4}"
DEMO="${DEMO:-$workspace/bazel-bin/mediapipe/examples/desktop/openvino_classification/openvino_classification}"
FRAMES="${FRAMES:-1000}"
read -r -a devices <<< "${DEVICES:-CPU GPU}"
read -r -a client_counts <<< "${CLIENT_COUNTS:-1 20}"
[[ "$FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo 'FRAMES must be a positive integer.' >&2; exit 2; }
for clients in "${client_counts[@]}"; do
    [[ "$clients" =~ ^[1-9][0-9]*$ ]] || { echo 'CLIENT_COUNTS must contain positive integers.' >&2; exit 2; }
done
for device in "${devices[@]}"; do
    [[ "$device" == CPU || "$device" == GPU ]] || { echo 'DEVICES must contain CPU and/or GPU.' >&2; exit 2; }
done
[[ -x "$DEMO" ]] || { echo "Demo executable not found: $DEMO" >&2; exit 1; }
for input in "$MODEL" "${MODEL%.xml}.bin" "$VIDEO"; do
    [[ -f "$input" ]] || { echo "Input not found: $input" >&2; exit 1; }
done
MODEL="$(realpath "$MODEL")"
VIDEO="$(realpath "$VIDEO")"
mkdir -p "$workspace/test-results"
OUTPUT_DIR="${OUTPUT_DIR:-$(mktemp -d "$workspace/test-results/resnet50-benchmark.XXXXXX")}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
printf 'device,sources,frames,seconds,aggregate_fps,fps_per_source,status\n' > "$OUTPUT_DIR/results.csv"
printf 'Model: %s\nVideo: %s\nFrames/source: %s\nResults: %s\n' "$MODEL" "$VIDEO" "$FRAMES" "$OUTPUT_DIR"

process=
stop_clients() {
    if [[ -n "$process" ]]; then
        kill "$process" 2>/dev/null || true
    fi
    wait || true
    exit 130
}
trap stop_clients INT TERM

benchmark_demo() {
    local device="$1" clients="$2"
    local output="$OUTPUT_DIR/${1}-${2}sources"
    local source count failed=0 total_frames=0 hint=THROUGHPUT
    local seconds=0 aggregate_fps= fps_per_source= summary
    [[ "$clients" != 1 ]] || hint=LATENCY
    mkdir -p "$output"
    local model_text="${MODEL//\\/\\\\}"
    model_text="${model_text//\"/\\\"}"
    model_text="${model_text//&/\\&}"
    model_text="${model_text//|/\\|}"
    sed -e "s|model_path:.*|model_path: \"$model_text\"|" \
        -e "s|device:.*|device: \"$device\"|" \
        -e "s/num_infer_requests: .*/num_infer_requests: $clients/" \
        -e "s/value: \"THROUGHPUT\"/value: \"$hint\"/" \
        "$script_dir/openvino_classification.pbtxt" > "$output/graph.pbtxt"
    printf 'Running %s with %s source(s)...\n' "$device" "$clients"
    "$DEMO" --calculator_graph_config_file="$output/graph.pbtxt" \
        --input_video_path="$VIDEO" \
        --input_width=224 --input_height=224 --input_nhwc=false \
        --max_frames="$FRAMES" --top_k=0 \
        --num_sources="$clients" --print_tensors=false \
        > "$output/run.log" 2>&1 &
    process="$!"
    wait "$process" || failed=1
    process=
    for ((source=0; source<clients; source++)); do
        count=$(awk -v source="$source" '
            $1 == "source" && $2 == source && $3 ~ /^frames=/ {
                sub(/^frames=/, "", $3); count += $3; summaries++
            }
            END { print summaries == 1 ? count+0 : 0 }' "$output/run.log")
        total_frames=$((total_frames + count))
        if [[ "$count" != "$FRAMES" ]]; then
            printf 'Source %s: expected %s frames, got %s: %s/run.log\n' \
                "$source" "$FRAMES" "$count" "$output" >&2
            failed=1
        fi
    done
    summary=$(awk '/^benchmark / {
        sub(/^seconds=/, "", $4); sub(/^aggregate_fps=/, "", $5)
        sub(/^fps_per_source=/, "", $6); print $4, $5, $6
    }' "$output/run.log")
    if [[ -n "$summary" ]]; then
        read -r seconds aggregate_fps fps_per_source <<< "$summary"
    else
        echo "Missing processing summary: $output/run.log" >&2
        failed=1
    fi
    count=$(grep -c 'OpenVINOSessionCalculator compiled ' "$output/run.log" || true)
    if [[ "$count" != 1 ]]; then
        echo "Expected one shared model compilation, got $count: $output/run.log" >&2
        failed=1
    fi
    awk -v device="$device" -v clients="$clients" -v frames="$total_frames" \
        -v seconds="$seconds" -v aggregate_fps="$aggregate_fps" \
        -v fps_per_source="$fps_per_source" -v failed="$failed" '
        BEGIN {
            if (failed) {
                printf "%s,%d,%d,%.3f,,,FAILED\n", device, clients, frames, seconds
            } else {
                printf "%s,%d,%d,%.3f,%.2f,%.2f,OK\n", device, clients, frames, seconds,
                    aggregate_fps, fps_per_source
            }
        }' | tee -a "$OUTPUT_DIR/results.csv"
    return "$failed"
}

failed=0
for device in "${devices[@]}"; do
    for clients in "${client_counts[@]}"; do
        benchmark_demo "$device" "$clients" || failed=1
    done
done
printf 'Results saved to %s/results.csv\n' "$OUTPUT_DIR"
exit "$failed"