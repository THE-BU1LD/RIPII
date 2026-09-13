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
| Retained v3/coupled models have measured but nonportable inference costs | engineering-verified | `world_v3_efficiency.json`; `world_global_coupling_v1_efficiency.json` | one CPU host; profiler FLOPs omit unsupported operators; not training compute matching |
| No legacy auxiliary objective improves reconstruction in the fixed objective study | negative development | `objective_study_v1/summary.json`; 39-run manifest | 3 seeds, 30 updates, synthetic; reconstruction is the primary outcome |
| Legacy objectives have materially unequal and conflicting gradients | mechanistic development | `objective_study_v1/gradient_diagnostics.json` | one deterministic batch per trained seed; diagnostic, not causal attribution |
| NRI multiscale study did not advance | negative external-simulator development | `nri_external_development_v1.json`; 185-file local manifest | 3 seeds/domain, 100 updates, compute-unmatched; not real-world or confirmatory |
| Version 0.2 mechanism defects are repaired | prospective engineering verification | semantic gradient/configuration tests | historical results predate these repairs and are not 0.2 evidence |
| Generic external trajectory interface executes | engineering-verified | manifest loader and end-to-end train/evaluate/benchmark tests | enables studies; does not supply a dataset or result |

Raw values remain in retained JSON/CSV and self-checksummed capsules; prose rounding is
never the authoritative source. These SHA-256 digests detect corruption but are not
authenticated author signatures. Historical artifacts are not rewritten after code changes.
