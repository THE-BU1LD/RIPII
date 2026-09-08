#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-runs/canonical_smoke}"
test ! -e "$output" || { echo "refusing existing output: $output" >&2; exit 2; }
"${PYTHON:-.venv/bin/python}" scripts/run_pipeline.py --config configs/mechanism_smoke.yaml --output-dir "$output"
