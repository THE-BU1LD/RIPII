# Evidence ledger

| Claim | Classification | Evidence | Boundary |
|---|---|---|---|
| Core models execute and differentiate | engineering-verified | full pytest suite | correctness, not usefulness |
| Legacy pilot v1 did not advance | negative, frozen local | `research/results/pilot_v1` | loss-balancer defect invalidates causal ranking |
| Corrected full RIPII is worse than no-VQ on pilot v2 | negative, frozen local | v2 summary: +0.061549 MSE | 3 seeds, 30 steps, synthetic |
| Pilot v2 codebooks are near collapse | negative, frozen local | effective fractions 0.148/0.182 | batch/task-specific diagnostic |
| World v3 multiscale loses graph OOD on 5/5 seeds | negative development | signed v2 capsule and full local manifest | one simulator/data seed; not confirmatory |
| Graph has lower mean v3 ID RMSE | development evidence | graph 0.0904 vs multiscale 0.0983 | descriptive rounding |
| Multiscale assignments remain active | mechanistic development | effective groups 2.77/4 | activity does not imply utility |
| Exact resume, manifests, fail-closed checks work | engineering-verified | tests and verification scripts | tested local CPU path |
| Novelty/superiority/external generalization | not established | `research/NOVELTY_AUDIT.md` | must not be claimed |
| External confirmatory protocol | not yet run | draft protocol | EXTERNAL_EXECUTION_REQUIRED |

Raw values remain in retained JSON/CSV and signed capsules; prose rounding is never the
authoritative source. Historical artifacts are not rewritten after code changes.
