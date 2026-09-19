#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output="${1:-runs/ripii_mr_development_v1}"

exec "$python_bin" scripts/run_ripii_mr_matrix.py --output "$output"
