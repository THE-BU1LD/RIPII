#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
input="${1:-research/results/development/world_v3_convergence_capsule_v2.json}"
output="${2:-runs/analysis/world_v3_convergence_statistics.json}"
"${PYTHON:-.venv/bin/python}" scripts/analyze_world.py "$input" --output "$output"
echo "analysis: $output"
