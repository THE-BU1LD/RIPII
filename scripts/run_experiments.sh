#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON:-.venv/bin/python}"
output="${1:-runs/world_development}"
shift || true

if test -e "$output"; then
  echo "refusing existing experiment output: $output" >&2
  exit 2
fi
"$python_bin" -m ripii.world benchmark --output "$output" "$@"
"$python_bin" -m ripii.world verify "$output"
echo "development experiment: $output"
