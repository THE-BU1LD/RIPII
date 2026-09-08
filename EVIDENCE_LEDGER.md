# Evidence ledger

| Claim | Classification | Evidence | Boundary |
|---|---|---|---|
| Core models execute and differentiate | engineering-verified | full pytest suite | correctness, not usefulness |
| Legacy pilot v1 did not advance | negative, frozen local | `research/results/pilot_v1` | loss-balancer defect invalidates causal ranking |
| Corrected full RIPII is worse than no-VQ on pilot v2 | negative, frozen local | v2 summary: +0.061549 MSE | 3 seeds, 30 steps, synthetic |
| Pilot v2 codebooks are near collapse | negative, frozen local | effective fractions 0.148/0.182 | batch/task-specific diagnostic |
| World v3 multiscale loses graph OOD on 5/5 seeds | negative development | self-checksummed v2 capsule and full local manifest | one simulator/data seed; not confirmatory |
| Graph has lower mean v3 ID RMSE | development evidence | graph 0.0904 vs multiscale 0.0983 | descriptive rounding |
| Multiscale assignments remain active | mechanistic development | effective groups 2.77/4 | activity does not imply utility |
| Exact resume, manifests, fail-closed checks work | engineering-verified | tests and verification scripts | tested local CPU path |
| Novelty/superiority/external generalization | not established | `research/NOVELTY_AUDIT.md` | must not be claimed |
| External confirmatory protocol | not yet run | draft protocol | EXTERNAL_EXECUTION_REQUIRED |
| Multiscale failure is not contact-specific | exploratory development negative | `world_v3_failure_analysis.json` | post-result localization; regimes overlap except free flight |
| Long-range coupling does not establish hierarchy-specific value | prospective development negative | `world_v4_coupling_capsule.json` | one simulator, 3 seeds, 100 updates |
| Longer coupled follow-up favors global pool in mean errors | exploratory development | `world_global_coupling_v1_capsule.json` | 3 new seeds, 300 updates; generic benchmark rule compares graph |

Raw values remain in retained JSON/CSV and self-checksummed capsules; prose rounding is
never the authoritative source. These SHA-256 digests detect corruption but are not
authenticated author signatures. Historical artifacts are not rewritten after code changes.
