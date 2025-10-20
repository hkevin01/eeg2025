#!/usr/bin/env bash
# Monitor GPU memory usage through rocm-smi and log CSV snapshots.

set -euo pipefail

print_help() {
  cat <<'EOF'
Usage: rocm_mem_watch.sh [options]

Options:
  -i, --interval SECONDS   Sampling interval in seconds (default: 5)
  -o, --output FILE        Output CSV file path (default: logs/rocm_mem_watch.csv)
  -d, --device ID          Limit monitoring to a specific device index (default: all)
  -c, --columns COLUMNS    Override CSV columns (default: built-in rocm-smi order)
  -h, --help               Show this message and exit

The script wraps "rocm-smi --csv" to capture VRAM usage and activity metrics at a
regular cadence. Press Ctrl+C to stop.
EOF
}

interval=5
output="logs/rocm_mem_watch.csv"
device=""
columns=""

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i|--interval)
        interval="$2"; shift 2 ;;
      -o|--output)
        output="$2"; shift 2 ;;
      -d|--device)
        device="$2"; shift 2 ;;
      -c|--columns)
        columns="$2"; shift 2 ;;
      -h|--help)
        print_help; exit 0 ;;
      *)
        echo "Unknown option: $1" >&2
        print_help
        exit 1 ;;
    esac
  done
}

parse_args "$@"

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "rocm-smi not found in PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"

header_written=false
trap 'echo "Stopping rocm_mem_watch" >&2; exit 0' INT TERM

while true; do
  timestamp="$(date --iso-8601=seconds)"
  cmd=(rocm-smi --showmemuse --showmeminfo vram vis_vram gtt --csv)
  if [[ -n "$device" ]]; then
    cmd+=(--device "$device")
  fi
  if [[ -n "$columns" ]]; then
    cmd+=(--columns "$columns")
  fi

  csv_output="$("${cmd[@]}")"
  if [[ "$header_written" = false ]]; then
    echo "timestamp,$csv_output" | head -n 1 >"$output"
    header_written=true
  fi
  echo "${timestamp},${csv_output}" >>"$output"

  sleep "$interval"
done
