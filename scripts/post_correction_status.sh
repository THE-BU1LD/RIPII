#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/runs/post_correction_v02}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
"$PYTHON_BIN" "$ROOT/scripts/run_post_correction.py" --output "$OUTPUT" --status
