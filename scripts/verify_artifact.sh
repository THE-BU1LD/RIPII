#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-.venv/bin/python}"
"$python_bin" scripts/verify_artifact.py --manifest research/results/pilot_v1/manifest.json --protocol research/protocols/pilot_v1.md --portable-summary
"$python_bin" scripts/verify_artifact.py --manifest research/results/pilot_v2/manifest.json --protocol research/protocols/pilot_v2.md --portable-summary
"$python_bin" -m ripii.world verify-capsule research/results/development/world_v3_convergence_capsule_v2.json
"$python_bin" scripts/run_coupling_study.py --verify-capsule research/results/development/world_v4_coupling_capsule.json
"$python_bin" -m ripii.world verify-capsule research/results/development/world_global_coupling_v1_capsule.json
"$python_bin" scripts/analyze_failures.py --verify-output research/results/development/world_v3_failure_analysis.json
"$python_bin" scripts/analyze_failures.py --verify-output research/results/development/world_global_coupling_v1_failure_analysis.json
"$python_bin" scripts/plan_power.py --verify-output research/planning/external_power_plan_v1.json
test -s research/results/development/world_v3_convergence_statistics.json
test -s research/results/development/world_global_coupling_v1_capsule.json
test -s audit/FINAL_AUDIT.md
test -s RESEARCH_TRUTH.md
test -s EVIDENCE_LEDGER.md
echo 'artifact verification: ok (portable evidence; omitted checkpoints not re-evaluated)'
