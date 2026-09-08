#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-.venv/bin/python}"
"$python_bin" scripts/verify_artifact.py --manifest research/results/pilot_v1/manifest.json --protocol research/protocols/pilot_v1.md --portable-summary
"$python_bin" scripts/verify_artifact.py --manifest research/results/pilot_v2/manifest.json --protocol research/protocols/pilot_v2.md --portable-summary
"$python_bin" -m ripii.world verify-capsule research/results/development/world_v3_convergence_capsule_v2.json
test -s research/results/development/world_v3_convergence_statistics.json
test -s audit/FINAL_AUDIT.md
test -s RESEARCH_TRUTH.md
test -s EVIDENCE_LEDGER.md
echo 'artifact verification: ok (portable evidence; omitted checkpoints not re-evaluated)'
