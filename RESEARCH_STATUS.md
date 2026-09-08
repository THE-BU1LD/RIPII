# Research status

Status: **experimental implementation; no publication claim**.

## Repository and verification status

The public source repository is `https://github.com/THE-BU1LD/RIPII`. Commit
`0dba137d49e9f827c6ca48724cb9f2106c81daf4` passed both hosted workflows:
[CI run 34230280332](https://github.com/THE-BU1LD/RIPII/actions/runs/34230280332)
and [quality run 34230280341](https://github.com/THE-BU1LD/RIPII/actions/runs/34230280341).
This verifies repository mechanics, not the scientific hypothesis. The method
remains a negative-result experimental prototype and still lacks an owner-approved
license.

Two frozen local pilots failed their predefined advancement conditions. The corrected
pilot v2 used three new seeds and exact retained initial states. Relative to `no_vq`,
base mean test reconstruction MSE was worse by 0.06155; relative to `no_structured`,
it was worse by 0.06169 and held-out probe accuracy was worse by 0.08772. These are
descriptive synthetic results, not inferential evidence.

## Verified

- The full RIPII architecture has forward, backward, and one-epoch tests.
- Training, evaluation, diagnostics, ablation, benchmark, and report CLIs execute.
- `configs/plumbing_smoke.yaml` is represented by the historical `configs/smoke.yaml`
  path and deliberately disables costly mechanisms after loading.
- `configs/mechanism_smoke.yaml` keeps projective refinement, graph refinement,
  hierarchical quantization, and latent action enabled.
- Pilot v1 retained 129 files and corrected pilot v2 retained 51; every declared hash
  and size passes full manifest verification in the retained local run directories.
  A clean repository checkout can separately verify each frozen protocol plus the
  committed JSON, CSV, and Markdown summaries with `--portable-summary`; that narrower
  mode explicitly reports the ignored run artifacts it did not verify.
- Evaluation is deterministic and train/validation/test splits are disjoint.
- A true plain autoencoder baseline is wired through configuration, training,
  evaluation, and benchmarking without structured RIPII mechanisms or unused
  parameter padding.
- The development benchmark automatically matches that baseline within 2% of full
  RIPII trainable parameters and records the actual mismatch.
- An isolated deterministic quantizer-qualification command records usage, assignment
  purity, reconstruction error, fixed gates, source hashes, and a development-only
  claim boundary.

## Not established

- Novelty relative to current representation-learning, graph, quantization, or
  multiscale-modeling literature.
- Improvement over an adequately powered, convergence/compute-matched baseline.
- Generalization beyond the bundled synthetic generator.
- Benefit from any individual RIPII mechanism.
- Statistical significance, scaling behavior, or independent reproduction.

The checked-in `runs/ripii_smoke` files are mutable historical development outputs.
They are not frozen evidence and must not be cited as results. Current evaluation fits
ridge probes on train features and scores them on a disjoint test split; the older
checked-in probe values were same-split diagnostics and remain non-evidence.

Pilot v1 exposed a loss-balancing defect: zero-weight objectives retained learned
uncertainty offsets and gradients, so its automatically generated total-loss ranking
is invalid. Global gradient clipping means the pilot must not be used for causal
component claims. The conservative no-advance decision remains recorded in
`research/results/pilot_v1/analysis.md`. The defect is fixed prospectively; the
manifested v1 files were not rewritten. The pilot also showed near-collapsed base
codebooks (mean coarse/fine perplexity 1.076/1.312 out of eight entries).

Pilot v2 ran after the fix and reproduced the core failure on new seeds. Base mean
effective-code fractions were 0.148 coarse and 0.182 fine, both below the frozen 0.25
gate, and base reconstruction was worse than `no_vq` on all three seeds. No current
result supports keeping the quantizer in the method.

The retained development-only checks from 2026-09-08 did not rescue the method. In a
single-seed, one-step parameter-matched comparison, RIPII used 33,380 trainable
parameters and reconstruction MSE 0.7279; the plain autoencoder used 33,048 parameters
(0.995% mismatch) and MSE 0.3774. This is a plumbing diagnostic, not an estimate of
converged performance. The isolated quantizer test passed code-usage and reconstruction
gates but failed both assignment-purity gates (0.75 coarse, 0.25 fine), so it did not
qualify. The exact development artifacts are under `research/results/development/`.

## Public-release gate

Before describing RIPII as a research release, add an owner-approved license and
authorship metadata, freeze and run an adequately powered convergence/compute-matched
protocol, evaluate on external data, and obtain independent review.
Until then, a public repository must be labeled **experimental prototype**.

## Object-state world-model implementation (2026-09-08)

A separate executable world-model path now provides a soft-disc dynamics simulator,
MLP/graph/Transformer/global-pool/multiscale predictors, fixed rollout losses, exact
CPU resume, masked variable object counts, continuous/FSQ/VQ bottlenecks, three
analytic references, and mechanism-specific collapse diagnostics. The global-pool
variant is a direct control for whether learned multiscale grouping adds value beyond
ordinary global context. The interactive Matplotlib workbench performs real inference
from trained checkpoints and compares its rollout with simulation under user-selected
interventions. Inputs are known object states; learned visual perception is not
implemented.

The first local world benchmark trained 12 models (four architectures × seeds 3, 7,
11) for 300 updates each. Its prospectively written development gate returned
**no_advance**: multiscale did not beat graph by 5% OOD with <=5% ID regression on
every seed. Its mean OOD errors were slightly lower, but two of three paired seeds
failed the rule. The initial report and all checkpoints are in `runs/world_v1`;
those run directories are ignored development outputs, not a new frozen paper pilot,
and they predate the global-pool control, explicit persistence/force baselines,
collapse diagnostics, and source snapshots. See `WORLD_MODEL.md` for the reproducible
commands and limitations. Existing pilot v1/v2 artifacts retain their original hashes
and conclusions.

A fresh five-architecture control run (`runs/world_v2_controls`, seeds 13, 17, 19)
also returned **no_advance** and passes its 124-file manifest check. Mean test position
RMSE was graph 0.1492, multiscale 0.1560, Transformer 0.1574, global-pool 0.1657,
and MLP 0.1939. Graph also had the lowest mean error in each of the three OOD
regimes. Multiscale beat the global-pool control on mean OOD error (0.2262 versus
0.2392), but the paired direction was inconsistent across seeds and this comparison
was not the predeclared advancement gate. Against graph, only seed 17 passed; seeds
13 and 19 failed. The grouping diagnostics remained active rather than numerically
collapsed (mean effective groups 3.09 of 4), but activity did not establish utility.
These are fixed-dataset, three-seed development results, not publication-grade
inference or external validation.

The stronger `runs/world_v3_convergence` study changed the dataset seed to 9091,
used five new initialization/minibatch seeds (23, 29, 31, 37, 43), raised the
training ceiling to 1,000 updates, and focused on graph, global-pool, and
multiscale. Its prospectively written protocol has SHA-256
`fb8c3a1ca56f68a848b32c3e012b170b3a171ef3e4b5aeed96129854f82c116f`,
and all 124 declared artifacts verify. The result is again **no_advance**. Mean
test position RMSE is graph 0.0904, global-pool 0.0920, and multiscale 0.0983.
Graph has the lowest mean error on every OOD split; multiscale's average OOD
relative improvement versus graph is -13.05% and it loses on all 5/5 paired
seeds. Against global-pool, multiscale has only a 1.58% mean OOD improvement,
loses 7.49% on mean ID relative error, and passes the descriptive 5% OOD/no-more-
than-5%-ID-regression check on 2/5 seeds. Its grouping remains active (2.77 mean
effective groups out of four), but again does not establish utility. This stronger
negative result argues against advancing the multiscale mechanism in its current
form. A signed compact copy of the protocol, full summary, report, and manifest is
retained at `research/results/development/world_v3_convergence_capsule_v2.json`; its
signature is
`ce4ffcfd9e5fcda6b07d3f6905ca9ea720f7b5623a761d06f8c0c540690ad370`.
The v2 capsule verifies the exact embedded byte count and SHA-256 of every retained
file; the earlier v1 capsule is preserved as superseded development provenance.
