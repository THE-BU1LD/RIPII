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
| Repaired RIPII 0.2 full objective fails its frozen advancement gate | negative development | `runs/post_correction_v02_parallel_20260913/study_summary.json`; 180/180 verified cells | five seeds, one synthetic generator; development-only, not confirmatory |
| Reconstruction+KL and the plain autoencoder beat repaired RIPII 0.2 on every paired seed | negative development | mean MSE: base 0.08410, reconstruction+KL 0.04687, plain AE 0.05573; exact paired sign-flip p=0.0625 for each comparison | n=5 cannot provide conventional two-sided significance; multiplicity-adjusted p=1.0 |
| The completed repaired-objective run is archived with content verification | reproducibility-verified | `output/research/post_correction_v02_parallel_20260913.tar.gz`; SHA-256 `1b3cb759e23221053d9d5843e849483fc966c49b412f62cd71f403ac931c6e68`; 2,183-file internal manifest | local archive; no independent replay or DOI |
| RIPII-MR fails its frozen advancement gate | negative synthetic development | `runs/ripii_mr_development_v1/report.md` and `summary.json`; decision `no_advance`; 75 learned-model runs; 0/60 paired comparisons pass | three synthetic data seeds with five nested model seeds; not external or confirmatory evidence |
| RIPII-MR has negative mean OOD improvement against every learned control | negative synthetic development | equivariant -0.1596; graph -0.7824; global pool -1.1909; legacy multiscale -1.2134 | descriptive means over 15 paired cells/control; cluster-level confirmation is not powered |
| RIPII-MR routing does not yet provide conditional computation | engineering limitation | `ripii/world/multiresolution.py`: restriction, coarse edges and coarse impulses are evaluated before the route gates their contribution | no compute-saving or accuracy-compute claim is supported |

Raw values remain in retained JSON/CSV and self-checksummed capsules; prose rounding is
never the authoritative source. These SHA-256 digests detect corruption but are not
authenticated author signatures. Historical artifacts are not rewritten after code changes.
