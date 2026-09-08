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

## Changes completed

Added mathematical and hypothesis specifications, evidence/checklist maps, novelty
review, adversarial reviews, negative-result manuscript, paired seed-level statistics,
external protocol boundary, canonical commands, and tests. Statistical analysis uses
five paired seeds and explicitly reports its low resolution. Speculative model changes
were not added because no retained observation justifies them.

## Remaining blockers

`EXTERNAL_EXECUTION_REQUIRED`: public datasets, frozen preprocessing and hashes,
compute/FLOP-matched convergence, at least ten prospectively chosen seeds or a justified
power design, independent reproduction, and checkpoint archival. Owner action is also
required for license and authorship metadata.

## Final verification record

- Full suite: **73 tests passed** (`./scripts/test.sh`).
- Targeted post-build regression: statistics, research-boundary, and world artifact/
  demo integration tests passed.
- Ruff: **pass**. Python compilation: **pass**. `git diff --check`: **pass**.
- Mechanism-enabled end-to-end smoke: **pass**, retained under ignored
  `runs/audit_final_smoke_20260908`; it is development plumbing, not evidence.
- Seed-level v3 analysis regeneration: **pass**, with exact sign-flip and Holm-adjusted
  results retained in `research/results/development/world_v3_convergence_statistics.json`.
- Pilot v1/v2 portable summaries: **pass** (3/129 and 3/51 declared files checked;
  omitted run files explicitly skipped). Signed v3 capsule: **pass**, signature
  `ce4ffcfd9e5fcda6b07d3f6905ca9ea720f7b5623a761d06f8c0c540690ad370`.
- Wheel and source distribution: **built successfully** with `uv build`.

The smoke emitted benign warnings when a single `runpy` process attempted to reset
PyTorch inter-op thread limits and when Matplotlib first built its font cache; neither
warning changed the successful outputs. No manuscript PDF was built because this
repository has only an evidence-limited Markdown negative-result draft.
