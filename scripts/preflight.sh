#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-.venv/bin/python}"
ruff_bin="${RUFF:-.venv/bin/ruff}"
"$python_bin" -c 'import numpy, torch, yaml; import ripii; print("preflight imports: ok")'
"$python_bin" -m compileall -q ripii scripts tests
"$ruff_bin" check ripii scripts tests
