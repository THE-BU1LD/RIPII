# Repository map

Audited: 2026-09-08. Evidence labels: conceptual, implemented,
engineering-verified, development evidence, frozen local pilot, external validation,
negative/falsified, and not yet run.

## Purpose and strongest defensible contribution

The repository contains two related but distinct experiments. The legacy RIPII path
tries to reconstruct synthetic vectors through a variational encoder, learned
projective refinements, latent graph processing, transformation-conditioned updates,
and hierarchical VQ. Its two frozen local pilots are negative. The newer world-model
path predicts object states in a deterministic 2-D soft-contact simulator using a
shared kinematic prior and MLP, graph, Transformer, global-pool, or learned multiscale
processors. Its strongest five-seed development study also rejects advancement of the
multiscale processor. The artifact's strongest contribution is therefore a reproducible
negative comparison and a compact, fail-closed experimental system—not a superior model.

## End-to-end map

| Stage | Canonical path | State / evidence |
|---|---|---|
| Question and hypotheses | `research/HYPOTHESES.md` | Frozen retrospectively for existing studies; new confirmatory work not run |
| Mathematics | `research/MATHEMATICAL_SPEC.md` | Reconstructed from code |
| Legacy synthetic data | `ripii/data/synthetic.py` | Implemented; synthetic only |
| Legacy model | `ripii/models/`, `ripii/utils/training.py` | Implemented and tested |
| Legacy experiments | `scripts/train.py`, `scripts/benchmark.py`, `scripts/evaluate.py` | Implemented; pilots v1/v2 retained |
| Object-state data | `ripii/world/physics.py` | Implemented deterministic simulator |
| World models | `ripii/world/models.py` | Five learned variants, three analytic controls |
| World training/selection | `ripii/world/experiment.py` | Validation-best selection; test generated after training |
| Statistics | `ripii/utils/statistics.py`, `scripts/analyze_world.py` | Seed-level paired descriptive/inferential diagnostics |
| Evidence | `research/results/` and ignored local `runs/` | Frozen legacy pilots; development world capsules |
| Paper path | Evidence-limited negative manuscript draft | Evidence is insufficient for a positive-method paper |
| Release/build | `pyproject.toml`, `.github/workflows/` | Package and CI paths implemented; no owner-approved license |

## Implementation truth

- Projective refinement, graph refinement, latent action, VQ, five world predictors,
  analytic controls, resume, manifests, and capsules execute.
- `configs/smoke.yaml` is intentionally plumbing-only. It is not mechanism evidence.
- `no_structured` is a component-removal control, while `plain_ae` is the true
  unstructured baseline.
- World-model inputs are privileged object states. There is no pixel encoder,
  learned object discovery, real-world dataset, or hardware evaluation.
- Equal optimizer updates and approximately matched parameters are implemented;
  equal FLOPs/convergence are not.
- No canonical path contains random/constant learned predictions, swallowed broad
  exceptions, `pass`, `NotImplementedError`, or hand-authored result tables.

## Evidence-producing paths

Frozen pilot summaries are regenerated from manifests by `scripts/verify_artifact.py`.
World experiments write a protocol before optimization, validation-select `best.pt`,
evaluate held-out splits only after all training, and hash artifacts. `scripts/analyze_world.py`
uses initialization/minibatch seed—not timesteps—as the paired unit. Figures are
diagnostic outputs; there is no manuscript table claiming publication-grade inference.

## Obsolete, historical, and dead paths

Pilot v1 is historically meaningful invalidated evidence and must remain immutable.
The v1 world capsule is superseded by v2 but retained for provenance. `.DS_Store`,
build products, caches, and local run directories are noncanonical generated debris;
they are ignored. There are no notebooks. The legacy vector path is maintained for
artifact verification but should not be extended without a new hypothesis.

## Critical dependencies and blockers

Python 3.10+, PyTorch, NumPy, PyYAML, tqdm, and Matplotlib are required. Pytest and
Ruff are development dependencies. External scientific completion requires an
owner-approved license/authorship record, public real-world data or learned perception,
adequately powered independent-dataset experiments, compute matching, and independent
reproduction.
