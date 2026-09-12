# Final audit

## Verdict

**EVIDENCE_PARTIAL.** The software is implementation-ready and locally reproducible,
but the scientific evidence is confined to synthetic development studies. Those
studies are consistently negative for the proposed structured mechanisms.

## P0 findings

Historical pilot v1 used inactive-objective uncertainty offsets and cannot support a
total-loss ranking; it is preserved and labeled. Current code fails closed on nonfinite
values, uses sample-weighted evaluation, disjoint splits, validation-best world
checkpoints, exact resume state, and manifest/capsule verification. No new P0 leakage
or result-fabrication path was found in canonical code.

## Scientific findings

The legacy model's many weakly identifiable auxiliary objectives do not demonstrate
useful structure. Pilot v2 shows worse reconstruction than quantizer bypass and full
mechanism removal, with near-collapsed codebooks. The strongest world study shows a
flat local graph outperforming learned multiscale grouping on every paired OOD seed.
The simplest supported interpretation is that local interaction bias is sufficient
for the present simulator and unconstrained grouping adds optimization/capacity burden.
Two long-range follow-ups also fail to establish hierarchy-specific value over global
pooling. Nonlocality alone is therefore not a supported rescue of the mechanism.

## Changes completed

Added mathematical and hypothesis specifications, evidence/checklist maps, novelty
review, adversarial reviews, negative-result manuscript, paired seed-level statistics,
external protocol boundary, canonical commands, and tests. Statistical analysis uses
five paired seeds and explicitly reports its low resolution. A targeted symmetric
long-range-force extension was implemented because it directly challenged the central
mechanism. It failed its prospective development rule. Failure-regime evaluation was
also implemented and run across all retained v3 checkpoints. The second pass added
validated/fingerprinted dataset adapters, immutable protocol validation, manifested run
states, self-checksummed analysis artifacts, practical-equivalence summaries, stability diagnostics,
and a repeatable machine-local profiler. The retained coupled checkpoints were profiled
under one CPU protocol. Repeated sessions changed latency ordering, so the result is
correctly bounded as load-sensitive and nonportable; profiler FLOPs remain explicitly
incomplete.

The final 39-run legacy objective study also completed without dropped cells. No single
auxiliary objective passed its paired 5% reconstruction-improvement rule; every addition
was worse on every seed, and the complete objective was worse by 26.4%/60.0%/89.5%.
This is short-budget development evidence and does not establish asymptotic behavior.
The signed gradient diagnostic additionally finds strong scale imbalance (invariance
mean weighted norm 6.04 versus reconstruction 1.32) and at least one negative seed for
63/78 objective pairs. It is diagnostic rather than causal evidence.

## Second hostile-audit result

The extension produced a boundary condition, not a rescue: global information improves
the long-range simulator relative to a local graph, but learned grouping does not
consistently improve upon global pooling. The interaction study returned `no_advance`,
and a separate 300-update follow-up found global pooling best in mean IID and every mean
OOD regime. No evidence category was upgraded.

## Remaining blockers

`EXTERNAL_EXECUTION_REQUIRED`: public datasets, frozen preprocessing and hashes,
compute/FLOP-matched convergence, the provisionally planned 18 paired seeds per dataset
(subject only to upward revision from external pilot variance), independent reproduction,
and checkpoint archival. Owner action is also required for license and authorship metadata.

## Final verification record

- Full suite: **106 tests passed** (`./scripts/test.sh`), including the post-cleanup
  cache-disabled rerun.
- Targeted post-build regression: statistics, research-boundary, and world artifact/
  demo integration tests passed.
- Ruff: **pass**. Python compilation: **pass**. `git diff --check`: **pass**.
- Mechanism-enabled end-to-end smoke: **pass**, retained under ignored
  `runs/audit_final_smoke_v2_20260908`; it is development plumbing, not evidence.
- Seed-level v3 analysis regeneration: **pass**, with exact sign-flip and Holm-adjusted
  results retained in `research/results/development/world_v3_convergence_statistics.json`.
- Pilot v1/v2 portable summaries: **pass** (3/129 and 3/51 declared files checked;
  omitted run files explicitly skipped). Self-checksummed v3 capsule: **pass**, digest field
  `ce4ffcfd9e5fcda6b07d3f6905ca9ea720f7b5623a761d06f8c0c540690ad370`.
- Wheel and source distribution: **built successfully** with `uv build`; the sdist was
  inspected and contains no ignored `research/results/**/runs/` subtree.

The canonical runner now isolates stages in subprocesses and gives Matplotlib a private
writable cache while suppressing only its benign first-use font-cache notice. The final
mechanism smoke had no thread/cache warnings. No manuscript PDF was built because this
repository has only an evidence-limited Markdown negative-result draft.
