# External dynamics comparison v1

Status: **draft, not frozen, not run — EXTERNAL_EXECUTION_REQUIRED**.

The question, H0/H1, primary effect threshold, controls, unit of analysis and decision
rule are specified in `../HYPOTHESES.md`. Before freezing, an owner must select at least
two public object-dynamics datasets, record version/hash/license and preprocessing,
define train/validation/OOD splits without inspecting test outcomes, fix at least ten
seeds, specify hardware and compute-matching measurement, and publish a SHA-256 digest
with an external timestamp. Any outcome observed before those fields are fixed is
development evidence and cannot be relabeled confirmatory.

Failures, non-finite runs, and exhausted budgets count as failures and remain in raw
artifacts. Model selection uses validation only. Raw per-seed predictions and metrics,
complete configurations, source commit/dirty state, dependency versions, hardware,
timings, checkpoints, and success/failure states are mandatory.
