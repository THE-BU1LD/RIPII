#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/runs/post_correction_v02}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
WORKERS="${RIPII_WORKERS:-5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
PID_FILE="$OUTPUT/overnight.pid"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "An overnight run is already active with PID $OLD_PID" >&2
    exit 1
  fi
fi
"$PYTHON_BIN" "$ROOT/scripts/run_post_correction.py" --output "$OUTPUT" \
  --workers "$WORKERS" --preflight
"$PYTHON_BIN" "$ROOT/scripts/run_post_correction.py" --output "$OUTPUT" \
  --workers "$WORKERS" --calibrate-only
"$PYTHON_BIN" "$ROOT/scripts/run_post_correction.py" --output "$OUTPUT" \
  --workers "$WORKERS" --duration-check --max-hours 24
# Complete and verify one real 600-step cell synchronously. The overnight process
# resumes from this cell instead of trusting short calibration runs alone.
"$PYTHON_BIN" "$ROOT/scripts/run_post_correction.py" --output "$OUTPUT" \
  --workers "$WORKERS" --stop-after-cells 1
cd "$ROOT"
nohup caffeinate -dimsu "$PYTHON_BIN" "$ROOT/scripts/run_post_correction.py" \
  --output "$OUTPUT" --workers "$WORKERS" >> "$OUTPUT/overnight.log" 2>&1 &
RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$PID_FILE"
echo "Started RIPII post-correction run (PID $RUN_PID)"
echo "Log: $OUTPUT/overnight.log"
echo "Status: $OUTPUT/launch.json"
echo "Check progress: $PYTHON_BIN $ROOT/scripts/run_post_correction.py --output $OUTPUT --status"
