# End-to-End Scientific and Reproducibility Audit of RIPII

## Executive verdict

**Decision: reject / not conference-ready.** The repository is unusually candid, substantially tested, and contains real implementations rather than a paper-only façade. Its strongest contribution today is a well-instrumented **negative-results research program**: on the included controlled simulators, neither the legacy structured autoencoder nor the newer learned multiscale world model demonstrates the claimed practical advantage. That negative conclusion is scientifically useful, but the evidence package is not yet a complete external-review artifact.

The strongest parts are the deterministic synthetic generator, real training and evaluation paths, mechanistic diagnostics, held-out split discipline, exact paired tests, source hashing, artifact manifests, pinned external-source record, 149 passing tests, lint/compile/build gates, and unusually honest claim boundaries. The current manuscript correctly avoids claiming a validated new method.

The weakest parts are decisive:

- The central positive idea—learned hierarchy or the full RIPII objective improves dynamics/reconstruction—has **not survived its own controls**. In the five-seed world-model study, the graph baseline beats multiscale on all five OOD suites. In a fresh 600-step seed-1009 legacy diagnostic, `base` reconstruction MSE was 0.06989 versus 0.04433 for the same architecture with the simple objective and 0.05028 for the plain autoencoder. The full objective also collapsed both 64-entry codebooks to effective perplexity approximately 1.
- The evidence remains almost entirely synthetic and simulator-internal. The “external” NRI study uses a second synthetic simulator, discards its graph labels, inserts constant radius/mass fields, supplies zero actions, and retains only a compact result capsule—not the raw data or trained checkpoints.
- The comparisons match parameter count and update count reasonably well, but not convergence, FLOPs, memory, wall-clock budget, or hyperparameter-search budget. Strong geometric/equivariant and modern simulator baselines are absent.
- The statistical studies are too small for conventional inference. With exact two-sided sign-flip tests, three paired seeds cannot yield a p-value below 0.25 and five cannot yield one below 0.0625. Holm correction across many post-correction contrasts cannot repair this lack of resolution.
- The manuscript is a short Markdown report, not a complete submission: no rendered paper, numbered equations, figures, result tables, author/affiliation/conflict statements, or owner-approved license.
- The attempted 180-cell post-correction overnight run failed on its first cell. Its two-point runtime calibration observed 60 steps completing faster than 20, clamped the inferred per-step cost to zero, and still approved the run. The original wrapper suppressed the child process error. This is an orchestration failure, even though the same cell and a fresh three-control diagnostic subsequently ran successfully.

**What is genuinely novel?** The exact combination of projective latent views, a latent graph, residual vector quantization, adaptive multi-objective weighting, and a learned multiscale dynamics hierarchy is locally distinctive. The broad scientific ingredients are not new: object-centric interaction networks, graph simulators, equivariant dynamics, multiscale mesh/message-passing simulators, latent relational inference, and discrete bottlenecks are established lines of work.[^1][^2][^3][^4][^5][^6][^7] The repository does not currently establish that its particular combination yields a new capability, superior scaling law, stronger invariance, or better predictive performance. The defensible contribution is therefore the controlled failure analysis and infrastructure, not a state-of-the-art model.

## Audit basis and verification record

This audit inspected the source tree, configuration files, protocols, manuscript, result summaries/capsules, artifact manifests, test suite, CI, packaging, run scripts, and generated outputs. It traced both implemented systems:

1. **Legacy structured autoencoder:** synthetic image generation → deterministic split → encoder → projective/graph/action/VQ mechanisms → decoder → adaptive objective → checkpoint → evaluation/probes/diagnostics.
2. **World model:** soft-disc simulator or NRI adapter → sequence split → one-step/rollout training → checkpoint → autoregressive evaluation → OOD/physics/failure/statistical summaries.

Checks executed on the audited working tree:

| Check | Result | Evidence |
|---|---:|---|
| Full tests | **149 passed** in 210.72 s | local `pytest` run |
| Ruff | **passed** | local `ruff check` |
| Python byte compilation | **passed** | `python -m compileall -q ripii scripts` |
| Patch whitespace | **passed** | `git diff --check` |
| Distribution build | **passed** after network-enabled dependency resolution | `/tmp/ripii-audit-dist/ripii-0.2.0.tar.gz`, wheel |
| Portable artifact verification | **passed for retained files** | `scripts/verify_artifact.sh` |
| Fresh legacy smoke pipeline | **passed** | `runs/audit_e2e_20260913/` |
| Fresh 600-step three-control run | **passed and full-manifest verified** | `runs/audit_post_correction_controls_seed1009_v2/` |
| Canonical post-correction overnight launch | **failed, 0/180 cells completed** | `runs/post_correction_v02/launch.json`, `overnight.log` |

Passing artifact verification means the retained files match their recorded digests; it does **not** mean all historical runs can be recomputed. The portable pilot manifests verify only 3/129, 3/51, and 3/201 retained artifacts respectively; run directories and checkpoints were intentionally omitted. Source-drift reports are expected because the current tree is newer than those snapshots.

The working tree was heavily modified before and during this audit. Historical hashes remain useful for identifying the code used by each retained study, but the checkout itself is not a clean, immutable release state. No existing research implementation was changed as part of this report.

## Scientific hypothesis, claimed contribution, and actual contribution

### Hypotheses

The repository contains two related but not identical hypotheses:

- **Representation hypothesis:** explicitly structured latent mechanisms—projective views, relational propagation, learned actions, hierarchy, and discrete codes—should improve reconstruction and yield identifiable, non-collapsed representations relative to simpler controls.
- **World-model hypothesis:** a learned soft multiscale grouping mechanism should improve long-horizon and OOD dynamics prediction over flat MLP, transformer, graph, and global-pooling controls, especially when interaction structure is hierarchical or globally coupled.

These hypotheses are testable. The present evidence rejects or fails to support both in the tested regimes.

### Claimed versus actual contribution

| Item | Claimed/intended | Evidence-supported conclusion |
|---|---|---|
| Projective hierarchy | structured coarse-to-fine latent views | Real differentiable subspaces, but no demonstrated semantic hierarchy or coarse-graining law |
| Full RIPII objective | jointly encourages useful structure | Worse reconstruction and probes than simpler controls in the fresh diagnostic; collapse persists |
| Multiscale world model | learned grouping should help OOD dynamics | Implemented, but graph wins every OOD suite in the main five-seed comparison |
| Physical structure | simulator-informed object dynamics | Real object states and interaction models; predictor is not symmetry-equivariant or conservation-preserving |
| External validation | transfer beyond the built-in simulator | Second synthetic NRI simulator only; negative and not fully retained |
| Reproducibility | auditable, deterministic evidence | Strong integrity tooling; incomplete end-to-end archival of raw data/checkpoints and a failed canonical overnight runner |

The manuscript’s negative framing is more defensible than the method branding. A serious paper should lead with the falsified hypotheses, controlled diagnostics, and lessons about objective interference/hierarchy—not imply validated “renormalization,” physical abstraction, or external generality.

## Component classification

The labels below use the requested taxonomy. A component can receive more than one label when implementation completeness and scientific adequacy differ.

### Core code

| Component/files | Classification | Evidence |
|---|---|---|
| `ripii/data/synthetic.py` | **complete/real**, **untested scientifically** | Generates actual rendered multi-shape data, deterministic labels and splits; unit-tested, but no empirical provenance, sensor model, or realism validation |
| `ripii/models/encoder.py`, `decoder.py`, `layers.py` | **complete/real** | Executed by training, checkpointing, smoke, and evaluation paths |
| `ripii/models/projective.py` | **complete/real**, **partial scientifically** | QR-orthonormal bases and learned mixing are real; “renormalization/coarse-graining” interpretation is not established |
| `ripii/models/graph.py`, `latent_action.py` | **complete/real**, **partial scientifically** | Differentiable graph/action mechanisms execute; no causal identifiability or transformation-law validation |
| `ripii/models/quantizer.py` | **complete/real**, **broken empirically** | Residual VQ and usage metrics are real; fresh base run used effectively 1/64 codes at both levels |
| `ripii/models/ripii.py` | **complete/real**, **broken as a competitive method** | Full forward/loss is implemented; 13-term objective underperforms controls and produces collapsed representations |
| `ripii/models/baselines.py`, `factory.py` | **complete/real**, **partial comparison** | Plain AE and mode factory are functional; cross-architecture “paired initialization” shares only compatible tensors |
| `ripii/utils/loss_balancer.py` | **complete/real**, **hardcoded shortcut**, **partial mathematically** | Implements learned log-variance weighting; objective scale can become negative and is not comparable across active-loss sets |
| `ripii/utils/metrics.py`, `statistics.py`, `power.py` | **complete/real** | Non-finite values fail closed; exact sign flips, bootstrap, Holm, and planning utilities are tested; tiny samples remain intrinsically underpowered |
| `ripii/utils/training.py`, `seed.py`, `reporting.py` | **complete/real**, **partial reproducibility** | Deterministic seeded training/checkpointing/reporting works; device-level determinism and resume equivalence are not comprehensively certified |
| `ripii/world/physics.py` | **complete/real**, **hardcoded shortcut** | Deterministic soft-disc physics, walls, drag, contact, action, and global coupling; coefficients/units are synthetic and empirically chosen |
| `ripii/world/models.py` | **complete/real**, **hardcoded shortcut**, **partial scientifically** | Five model variants execute; dense all-pairs interactions, radius 0.6, and output scales 0.5/0.05 are hardcoded; no E(2) equivariance |
| `ripii/world/data.py`, `protocol.py` | **complete/real** | Masked variable-object sequences and fixed OOD suites are implemented; all suites share the same generator and laws |
| `ripii/world/experiment.py` | **complete/real**, **partial comparison** | Training, validation, rollout evaluation, checkpoints, and metrics are real; budget matching and physical metrics are incomplete |
| `ripii/world/nri_data.py` | **complete/real adapter**, **placeholder/stub/mock semantics** | Real NRI arrays are loaded; radius=0.04, mass=1, and action=0 are inserted placeholders, and NRI edge labels are ignored |
| `ripii/world/external_data.py`, `external_benchmark.py` | **complete/real API**, **scaffold as evidence** | Strict adapter/benchmark and tests exist, but no genuinely external empirical dataset is supplied or reported |
| `ripii/world/inference.py`, `cli.py`, `demo.py`, `__main__.py` | **complete/real**, **partial productization** | Usable checkpoint inference and CLI paths; no compatibility matrix, serving contract, or calibrated uncertainty |
| `ripii/world/profiling.py`, `failure_analysis.py`, `run_status.py` | **complete/real**, **partial** | Profiling and diagnostics execute; status reporting does not robustly establish launcher/PID health or recover failed cells |

### Experiment and automation code

| Component/files | Classification | Evidence |
|---|---|---|
| `scripts/train.py`, `evaluate.py`, `diagnostics.py`, `benchmark.py` | **complete/real**, **partial robustness** | Fresh 600-step run and manifest verification succeed; original benchmark error reporting hid child output until corrected |
| `scripts/run_pipeline.py`, `run_suite.py`, `run_smoke.sh`, `run_experiments.sh` | **complete/real** | Smoke path completed raw generation through plots; full-scale runtime/failure recovery is not exercised in CI |
| `scripts/run_coupling_study.py`, `run_nri_external_study.py`, analysis scripts | **complete/real**, **untested at publication scale** | Produce retained development evidence; short runs and tiny seed counts do not support confirmatory claims |
| `scripts/run_post_correction.py`, `start_post_correction_overnight.sh`, `post_correction_status.sh` | **broken**, **hardcoded shortcut** | First canonical cell failed; 20/60-step timing inversion was clamped to zero slope; fixed five seeds, 36 modes, 600 steps, and 15% buffer; no cell resume/retry |
| `scripts/build_paper.sh` | **scaffold** | Only asserts `paper/MANUSCRIPT.md` is nonempty and explicitly produces no PDF/TeX artifact |
| `scripts/clean_generated.py` | **complete/real**, **untested operationally** | Narrow generated-file cleanup is tested, but destructive use is not part of the audited run |
| Configs in `configs/` | **complete/real**, some **hardcoded shortcut** | Parsed and used; synthetic constants, fixed loss weights, small horizons, and sentinel output paths are research choices rather than tuned/validated defaults |

### Evidence, paper, tests, and repository

| Component/files | Classification | Evidence |
|---|---|---|
| `research/results/pilot_v1/` | **partial**, **scientifically invalid for total-loss ranking** | Historical adaptive-weight bug allowed zero-weight terms to contribute log-variance terms; only compact artifacts retained |
| `research/results/pilot_v2/` | **complete/real development evidence**, **partial reproducibility** | Honest negative result and verified summary; checkpoints/run directories absent |
| World v3/coupling/NRI capsules | **complete/real retained summaries**, **partial reproducibility** | Digests verify retained files; cannot rerun inference from capsules alone |
| `research/results/development/objective_study_v1/` | **complete/real diagnostic**, **untested at convergence** | Three seeds × 30 updates; useful gradient/objective screen, not final evidence |
| `research/protocols/*` | **complete/real records**, some **partial** | Frozen digests and amendments are strong; post-correction protocol has no completed result and external confirmatory protocol remains future work |
| `research/HYPOTHESES.md`, `MATHEMATICAL_SPEC.md`, `NOVELTY_AUDIT.md` | **complete/real documentation**, **partial theory** | Clear claims and equations; no theorem establishing hierarchy benefit, identifiability, consistency, or stability |
| `paper/MANUSCRIPT.md` | **partial**, **scaffold for submission** | Honest negative narrative but no standard paper apparatus, figures/tables, exhaustive related work, or rendered artifact |
| `audit/FINAL_AUDIT.md`, older reviewer/checklist files | **partial**, partly **unused/dead/stale** | Preserve historical review, but counts/status are stale (for example 106 versus 149 current tests) and should not be treated as current truth |
| `tests/` | **complete/real engineering suite**, **partial scientific validation** | 149 tests pass; excellent shape/semantics/CLI regression coverage, but no coverage threshold, type-check gate, long-run statistical golden tests, or independent reference implementation |
| `.github/workflows/*`, Dependabot | **complete/real**, **partial environment fidelity** | CI, quality, security, CodeQL, and pinned actions exist; CI installs editable requirements instead of reproducing `uv.lock` exactly |
| `pyproject.toml`, `uv.lock` | **complete/real** | Package builds; broad lower bounds remain, and the lock is not the CI authority |
| License/authorship/data cards/model cards | **missing** | No owner-approved license; no complete authorship, conflict, model card, or dataset card |
| Canonical pseudocode or executable stubs | **missing by design / none found** | No meaningful `NotImplementedError`, TODO body, or pass-only canonical implementation was found outside textual/config placeholders |

## Mathematics and scientific validity

### Projective views

For a raw basis matrix \(R\), the code computes a QR factorization and uses an orthonormal basis \(B\). The projector

\[
P = BB^\top
\]

is symmetric and idempotent because \(B^\top B=I\), so \(P^\top=P\) and \(P^2=P\). This part is mathematically sound, modulo finite-precision QR and sign ambiguity. Regularizing the *raw* basis orthogonality is an optimization preference, not necessary for the deployed QR basis to be orthonormal.

However, the model mixes learned bases and then re-orthogonalizes. In general,

\[
\operatorname{qf}\!\left(\sum_k \alpha_k B_k\right)
\]

is not equivalent to a convex mixture of projectors \(\sum_k\alpha_k B_kB_k^\top\), nor does it trace a geodesic or implement a renormalization semigroup. Calling it projective attention is fair; calling it learned renormalization requires evidence that is absent.

### Adaptive objective

Each active loss is transformed approximately as

\[
J_i(s_i)=w_i e^{-s_i}L_i+s_i.
\]

For fixed positive \(w_iL_i\), setting the derivative to zero gives

\[
s_i^*=\log(w_iL_i),\qquad J_i(s_i^*)=1+\log(w_iL_i).
\]

Therefore an apparently “better” total can become negative whenever enough scaled losses are small; totals from models with different active terms are not comparable. This also explains why learned log variances can obscure objective interference. The form is a legitimate uncertainty-style scalarization under particular likelihood assumptions, but the repository does not derive those assumptions for correlation, topology, entropy, geodesic, moment, and identity penalties. Those heterogeneous terms do not automatically share the probabilistic interpretation needed to justify a single formula.

The view-matching losses detach one target branch. That can be a reasonable stop-gradient design, but it makes the objective asymmetric and changes the stationary points. There is no derivation showing that the chosen direction estimates an invariant/equivariant representation rather than simply fitting the teacher branch.

### Quantization and collapse

The residual VQ path uses a standard straight-through construction, so gradients reach the encoder while codebook/commitment losses update the discrete bottleneck. The implementation is real. The empirical state is not healthy: in the fresh 600-step `base` run, coarse and fine perplexities were both 1.000001 with usage 0.015625, i.e. approximately one active entry of 64. This violates the intended multilevel discrete representation regardless of reconstruction quality. The simple-objective control had materially better—but still limited—perplexity (7.06 and 5.35) and usage (0.144), implicating objective interaction rather than absence of a quantizer implementation.

### Simulator dynamics and conservation

The contact force is applied pairwise with equal and opposite updates. The global harmonic coupling is likewise constructed symmetrically, so internal pair forces cancel and conserve total momentum in isolation. Walls, drag, and applied actions are external/non-conservative terms, so total momentum should not be expected to remain constant in their presence. A single aggregate “momentum drift” metric is therefore scientifically ambiguous unless conditioned on force regime and wall contacts.

Semi-implicit Euler with four substeps is a reasonable simple integrator, not a guarantee of energy accuracy. No convergence study over time step/substeps is reported. Spring stiffness, damping, drag, radii, mass ranges, and arena units are synthetic constants with no dimensional calibration. Penetration penalties and discontinuous contact onset can make gradients/trajectories sensitive near collision boundaries.

### Predictor update and dimensional assumptions

The learned world model uses

\[
v_{t+1}=v_t+0.5\tanh(\Delta v),\qquad
x_{t+1}=x_t+\Delta t\,v_{t+1}+0.05\tanh(\Delta x).
\]

The coefficients 0.5 and 0.05 are fixed output scales. They are dimensionally meaningful only relative to the implicit dataset normalization, yet state features with different units and ranges are not standardized. The residual position term can bypass the velocity-mediated dynamics, weakening physical interpretability. Radius and mass are copied exactly, which is appropriate for immutable properties but also makes property-drift metrics trivially favorable.

The architecture uses absolute positions and generic MLPs. It is permutation-equivariant over valid objects, but it is not guaranteed translation-, rotation-, or reflection-equivariant. The dense pair tensor is constructed even when a local mask is later applied, so graph and multiscale variants retain \(O(N^2h)\) pair memory/compute. The configured object cap (at most 16 in current studies) prevents any claim of demonstrated large-system scaling.

### Losses, gradients, and rollout

World-model training uses a four-step rollout loss with fixed coordinate weights 4, 4, 1, 1; evaluation reaches 32 steps. This is a legitimate truncated objective but creates exposure-bias and long-horizon mismatch. Validation selects position RMSE plus 0.25 velocity RMSE, which is not the same target as the weighted training loss or every reported OOD metric.

Masked RMSE aggregation is correctly normalized over valid objects, time, and the two spatial coordinates. The implementation also reports horizon, quantiles, worst scene, arena violations, property drift, and momentum drift. Missing scientific metrics include energy/work balance by force regime, collision/contact error, distributional trajectory distance, calibration/uncertainty, stability beyond the training horizon, and symmetry tests.

### Statistical validity

The paired exact sign-flip implementation is appropriate for small paired samples and the bootstrap intervals are useful descriptive summaries. The limitation is sample size, not code correctness. With \(n\) nonzero paired differences, the smallest possible two-sided exact sign-flip p-value is \(2/2^n\): 0.25 for \(n=3\), 0.0625 for \(n=5\), and 0.0078125 for \(n=8\). Thus the existing three- and five-seed studies cannot establish conventional two-sided significance even under unanimous effects; Holm correction across many outcomes only raises thresholds further.

The “every seed must improve by at least 5%” gate is a conservative engineering decision rule, not an estimator of generalization probability. The 18-seed external power plan was derived from internal synthetic effects and normal approximations; it should be treated as provisional and recomputed from an external pilot without reducing the preregistered minimum opportunistically.

## Model, algorithm, and implementation agreement

The implementation broadly matches the repository’s current mathematical specification. There is no evidence that the main methods are pseudocode or mocked. The mismatch is between **mechanism names/interpretations** and demonstrated properties:

- “Hierarchy” is a learned soft assignment and coarse interaction block, not validated scale separation.
- “Projective renormalization” is a learned orthonormal subspace mechanism without an RG flow, fixed point, universality argument, or semigroup consistency.
- “Equivariance” is encouraged through a finite sampled loss, not guaranteed architecturally.
- “Physics” is represented by object state and synthetic force generation, but the predictor does not enforce known symmetries or conservation.
- “External” currently means a different synthetic simulator, not empirical or community benchmark diversity.

Checkpoint saving/loading, best-model selection, inference, manifest creation, and evaluation are implemented. Numerical failure checks are stronger than average: non-finite metrics fail closed. Missing are automatic mixed-precision validation, gradient-norm/clipping sensitivity, optimizer/scheduler ablations, resume-bitwise-equivalence tests, multi-device determinism, and an independent implementation of critical metrics.

## Data audit

### Built-in rendered-shape data

Provenance is fully internal and deterministic. This avoids external licensing and contamination concerns, but it also makes the task narrow. The split is deterministic and held-out probes fit on train features then score the test set once. No obvious train/test leakage was found in that path.

Risks:

- Low-level rendering artifacts may make labels decodable without learning the intended abstraction.
- Train and test share the same generator family; distribution shift is limited to sampled factors.
- There is no image corruption, background diversity, anti-aliasing study, intervention test, or causal factor identifiability analysis.
- The labels and simulator parameters are generated together, so impressive probe scores would still not demonstrate real-world abstraction.

### Built-in world simulator

The default in-distribution training set uses 2–4 objects. OOD suites increase to 5–8 objects, hold out larger/heavier compositions, or alter velocity; coupling studies add a global force. These are meaningful controlled interventions but retain the same codebase, integrator, state schema, and much of the parameter support. A model can exploit generator-specific regularities.

Actions are piecewise-constant random chunks rather than a behavioral/control-policy distribution. No action-policy shift is tested. Only one simulator-data seed underlies some multi-model comparisons, so paired model seeds do not capture uncertainty in dataset generation.

### NRI and external adapters

The NRI pipeline pins an upstream source revision and records preparation. That is good provenance. However, the checkout does not include the raw arrays. The adapter rescales positions/velocities by a box size of five, inserts radius 0.04 and mass 1.0, sets actions to zero, and does not use NRI edge labels. Those choices must be stated as representation padding, not observed physics.

NRI remains synthetic and small. Stronger external validation should include at least one established physical-simulation benchmark with documented splits and ideally one noisy empirical trajectory dataset. Candidate benchmark ecosystems include LagrangeBench for particle-based simulation and The Well for diverse spatiotemporal physics.[^8][^9]

No dataset card currently records per-field units, ranges, missingness, licenses, checksums, generator versions, split hashes, or ethical/privacy assessment. The built-in generators have no human data risk, but external adapters need explicit license and terms checks.

## Experiment and evaluation audit

### Legacy model

The historical pilot v1 total-loss comparison is invalid because inactive losses could still contribute learned log-variance terms. Pilot v2 repaired that issue but remained negative and collapse-prone. The objective study isolates auxiliary additions, which is the right diagnostic structure, yet three seeds at 30 updates measure early optimization rather than converged model quality.

The fresh audit control used the current post-correction configuration, seed 1009, and 600 steps:

| Mode | Recon MSE ↓ | Held-out probe ↑ | Coarse/fine perplexity | Usage | Parameters |
|---|---:|---:|---:|---:|---:|
| Full `base` | 0.069890 | 0.1173 | 1.000 / 1.000 | 0.0156 | 1,812,465 |
| `simple_objective` | **0.044335** | 0.7997 | 7.057 / 5.354 | 0.1436 | 1,812,465 |
| `plain_ae` | 0.050285 | **0.9886** | N/A | N/A | 1,812,834 |

The simple objective lowers MSE by 36.6% relative to `base`; the plain AE lowers it by 28.1%. This is a single-seed development diagnostic, not an inferential result, but it is a strong falsification signal because the same-architecture control isolates the auxiliary objective. The full 180-cell protocol is still needed to localize which additions are harmful.

### World models

The main world v3 study is more compelling: five model seeds, fixed data/protocol, several OOD suites, paired comparisons, and retained statistics. It reports graph as better than multiscale on all five OOD suites. The global-coupling follow-up and NRI development study also fail to establish a multiscale advantage. The manuscript is correct to report this as a negative result.

Fairness limitations:

- Parameter counts and update counts are controlled; actual convergence, FLOPs, memory, wall time, and tuning budget are not.
- The transformer, graph, global-pool, and multiscale variants do not receive architecture-specific optimization sweeps.
- Dense all-pairs construction makes the “local” graph compute comparison misleading at larger N.
- The strongest relevant baselines are missing: E(n)-equivariant GNNs,[^4] noise-trained Graph Network Simulators,[^2] constraint-based simulators,[^10] multiscale MeshGraphNets/BSMS-GNN,[^5][^6] and energy/Lagrangian approaches.[^11]
- Four-step teacher-forced/autoregressive training and 32-step evaluation may favor some architectures differently; horizon-matched training and scheduled/noise rollouts are absent.
- No scaling sweep establishes the interaction between object count, receptive field, hierarchy depth, memory, and error.

### Metrics and conclusions

The primary trajectory RMSE implementation is not obviously wrong. The greater problem is construct validity: lower coordinate RMSE alone does not demonstrate physically correct dynamics or meaningful hierarchy. Momentum drift is not interpretable without separating closed, forced, drag, and wall-contact regimes. Property drift is near-trivial because properties are copied. Probe accuracy can be high for a collapsed or shortcut representation and needs calibration against label balance and nonlinear probes.

Every major current paper claim is consistent with retained summaries **because the manuscript is framed negatively**. Any positive claim stronger than “the mechanism is implemented and tested” would not be supported. The paper should include the failed audit run and fresh single-seed diagnostic as development evidence only, not silently promote them to confirmatory results.

## End-to-end reproducibility trace

| Stage | Legacy path | World/NRI path | Verdict |
|---|---|---|---|
| Raw data | Generated inside `ripii/data/synthetic.py` | Generated by `physics.py`; NRI prepared from pinned upstream | **complete/real** internally; NRI raw files **missing** |
| Preprocessing | Tensor rendering/normalization in generator/config | State packing/masks; NRI scaling and padded fields | **complete/real**; NRI semantics **partial** |
| Split | Deterministic 70/15/15 with held-out probe discipline | Protocol-defined train/validation/OOD suites | **complete/real**; split hashes/data cards **missing** |
| Training | `scripts/train.py`, benchmark modes | `experiment.py` and study runners | **complete/real**; long-run recovery **broken/partial** |
| Checkpoint | best/latest/final `.pt` | best/final run checkpoints | **complete/real** during runs; historical checkpoints **missing** from portable archive |
| Inference | `scripts/evaluate.py` | `world/inference.py`, CLI | **complete/real** for available checkpoints |
| Evaluation | recon/probes/mechanism diagnostics | rollout/OOD/physics diagnostics | **complete/real**, scientifically **partial** |
| Plots/tables | diagnostic PNG/JSON; summary scripts | analysis/capsule summaries | **complete/real** tooling; paper integration **missing** |
| Paper result | manually summarized from retained JSON/Markdown | manually summarized from capsules | **partial**; no single command rebuilds a rendered paper from raw runs |

The fresh `scripts/run_smoke.sh runs/audit_e2e_20260913` run completed all legacy stages and produced `best.pt`, `final.pt`, `eval.json`, `diagnostics.json`, rows, and five plots. It also yielded feature and structural effective rank approximately 1.0 and latent cosine mean 0.9966, correctly exposing smoke-scale collapse rather than hiding it.

The canonical overnight path did not complete. `runs/post_correction_v02/launch.json` records `status: failed` after the first `base` event. Calibration measured 58.90 s for 20 steps and 44.48 s for 60 steps, then used

\[
\max\left(0,\frac{44.48-58.90}{60-20}\right)=0
\]

seconds per step and estimated every 600-step cell from fixed overhead alone. The estimate happened to be of a plausible order for one later local run, but the estimator is mathematically non-informative under that timing inversion and must reject or repeat calibration. The failed cell subsequently ran successfully in isolation, and a fresh three-control benchmark plus 19/19 full-manifest verification also succeeded. The best diagnosis is therefore orchestration/resource/observability failure, not a deterministic training-code failure.

## Repository engineering audit

Engineering quality is above the scientific-readiness level:

- Clear package boundaries, CLIs, configs, source hashes, manifests, explicit evidence status, pinned GitHub Actions, Dependabot, CodeQL, and failure-closed metrics.
- Good regression tests for shapes, semantics, ablations, statistics, artifacts, NRI, inference, and protocol boundaries.
- The package builds successfully and includes relevant docs/research records.

Remaining engineering debt:

- CI does not install exactly from `uv.lock`, so local and CI environments can diverge.
- No coverage threshold, static type checker, performance regression gate, or long-run integration job.
- No robust per-cell state machine, atomic completion marker, resume, retry policy, heartbeat/liveness check, disk-space guard, or captured stderr file for overnight studies.
- The system volume had roughly 2.6 GiB free during audit while the research volume had ample space; caches/temp directories can still make long runs fragile.
- Historical audit/checklist documents are stale and compete with newer truth documents.
- Generated failed/incomplete audit directories are not catalogued consistently.
- No license, citation metadata, release tag/checksum bundle, software bill of materials, or archival DOI.

## Critical findings

| Severity | File/component | Problem | Evidence | Scientific impact | Exact fix |
|---|---|---|---|---|---|
| P0 | `ripii/models/ripii.py`, post-correction results | Full method currently loses to simpler controls and collapses | Fresh seed-1009 600-step MSE 0.06989 vs 0.04433; both VQ perplexities ≈1 | Central positive method claim is contradicted | Complete preregistered objective/ablation sweep; redesign or remove harmful terms; require non-collapse and held-out gain gates |
| P0 | `scripts/run_post_correction.py` | Runtime calibration accepts an impossible/noisy negative slope | 60-step timing <20-step timing; slope clamped to 0 | Overnight feasibility check is invalid | Use repeated warm runs at ≥3 horizons, robust regression/upper confidence bound, and reject non-monotone calibration |
| P0 | `runs/post_correction_v02/`, launcher/status scripts | Canonical study failed at cell 1 and is not resumable | `status: failed`, 0/180; original child error hidden | Planned evidence does not exist | Atomic cell manifests, stderr capture, PID/liveness, retry/resume, disk guard, and a preflight full-length sentinel cell |
| P0 | Repository root/paper | No owner-approved license or submission metadata | No license; manuscript lacks authors/affiliations/conflicts | Cannot safely redistribute or submit as a complete artifact | Owner selects license; add `LICENSE`, citation metadata, author contribution/conflict/funding statements |
| P1 | `paper/MANUSCRIPT.md`, `scripts/build_paper.sh` | No reproducible rendered paper | Build script only checks nonempty Markdown | Reviewers cannot verify claims/figures/tables | Add Quarto/LaTeX source; generate tables/figures from immutable summaries; CI-build PDF |
| P1 | Historical manifests/capsules | Portable artifacts omit raw runs/checkpoints | Majority of manifest entries skipped; NRI raw data absent | Results are integrity-checkable but not recomputable | Archive checkpoints, split indices, raw/prepared checksums, logs, environment lock, and rerun command in a versioned release/DOI |
| P1 | World comparisons | Missing strongest baselines and budget fairness | No EGNN/GNS/constraint/MGN/BSMS/energy model; updates only matched | Negative/positive ranking may be architecture- or tuning-dependent | Add modern baselines with equal search protocol; report params, FLOPs, peak memory, time, and convergence |
| P1 | `world/models.py`, `physics.py` | Predictor lacks physical symmetries and uses arbitrary scales | Absolute coordinates; 0.5/0.05 residual scales; no feature normalization | Weak physical inductive bias and brittle transfer | Normalize with train statistics; add relative-coordinate/equivariant baseline; ablate scales and residual bypass |
| P1 | Existing studies/statistics | Seed counts cannot support inferential claims | Exact p minima 0.25 and 0.0625 | Confidence/significance language would be invalid | Run powered external protocol; report effect sizes and CIs; label current studies developmental |
| P1 | `world/nri_data.py` | “External” state fields are fabricated constants and graph labels unused | radius/mass/actions inserted; edges dropped | Cross-domain claim is narrower than it appears | Explicit adapter ablation; consume relevant relation labels or remove relational claim; add real benchmark/data |
| P1 | World loss/evaluation | Four-step training versus 32-step evaluation; incomplete physics metrics | Protocol/code inspection | Long-horizon stability not directly optimized | Horizon/noise curriculum ablation; energy/contact/closed-system momentum and stability metrics |
| P2 | `world/models.py` | Dense all-pairs compute even for local model | Pair tensor materialized before mask | No demonstrated scaling benefit | Sparse neighbor lists/segment reductions; scaling study to realistic N |
| P2 | `utils/loss_balancer.py` | Objective totals are incomparable and probabilistic basis is unproved | Analytic optimum `1+log(wL)` | Misleading training/model selection | Log raw losses and gradient contributions; select on fixed external metric; derive or replace scalarization |
| P2 | CI/environment | Lockfile is not CI authority; no types/coverage/perf gates | Workflow inspection | Reproducibility and refactor safety gaps | Install with frozen lock; add pyright/mypy, coverage threshold, benchmark budget |
| P2 | Synthetic data | Same-generator evaluation permits shortcuts | All core splits derive from internal generators | Limited external validity | Rendering interventions, generator holdout, noisy/empirical data, dataset cards |
| P3 | Audit/docs/generated runs | Multiple stale truth documents and uncatalogued failures | Old audit says 106 tests; current is 149 | Reviewer confusion | Maintain one dated status index; mark superseded documents; register failed runs |

## Missing research

### Experiments

1. Complete the preregistered 36-mode × five-seed post-correction sweep with reliable resumption and immutable outputs.
2. Convergence-matched and compute-matched comparisons, including identical tuning budgets and learning-curve area, not only final equal-update scores.
3. Strong baselines: EGNN or comparable E(2)-equivariant model, noise-trained GNS, constraint projection, MeshGraphNet/multiscale and BSMS-GNN-style hierarchy, and an energy/Lagrangian model.[^2][^4][^5][^6][^10][^11]
4. Scaling sweeps over object count, density, interaction radius, hierarchy depth/groups, rollout horizon, and memory/time.
5. Symmetry interventions: translation, rotation, reflection, permutation, and coordinate-unit rescaling.
6. Integrator-resolution studies over `dt` and substeps to separate model error from simulator discretization error.
7. VQ rescue studies: EMA codebooks, reset/dead-code policy, entropy/diversity schedules, residual-depth ablation, and finite scalar quantization as a low-collapse alternative.[^7]
8. One genuinely external, licensed benchmark and preferably empirical/noisy trajectories; retain raw checksums and checkpoints.
9. Failure-mode stratification by collision count, wall contact, density, speed, mass/radius extremes, and force regime.
10. Long-horizon stability and uncertainty/calibration evaluation beyond 32 steps.

### Theory and mathematics

1. Define what “hierarchy” must satisfy: scale separation, assignment stability, compositional consistency, or improved approximation/complexity bounds.
2. Replace renormalization language unless a coarse-graining operator, flow, consistency relation, and measurable fixed-point/scaling behavior are defined.
3. Derive every auxiliary loss from an estimand or constraint and analyze incompatible gradients/stationary points.
4. Establish symmetry properties of each architecture and test approximate equivariance numerically.
5. Analyze stability/Lipschitz behavior of the autoregressive residual update and accumulated error.
6. Separate conservation claims by closed/open system and derive expected momentum/energy changes under actions, drag, contacts, and walls.
7. Predefine a primary estimand, smallest effect of interest, multiplicity family, and power analysis based on appropriate external-pilot variance.

### Data, documentation, and literature

1. Dataset cards with provenance, license, schema, units, ranges, splits, hashes, preprocessing, and known artifacts.
2. Model cards for every released checkpoint and explicit intended/non-intended use.
3. A modern related-work comparison table covering interaction networks, GNS, equivariant GNNs/operators, constraint-based simulation, multiscale GNNs, NRI, and physical benchmark suites.[^1][^2][^3][^4][^5][^6][^8][^10][^12]
4. A reproducible figure/table pipeline and archival release with DOI.

## Prioritized improvement checklist

Every item below names **what**, **why**, **how**, **where**, and **verification** explicitly.

### P0 — scientifically invalid or broken

- [ ] **Establish whether the full legacy method is viable.** **WHAT:** the full objective underperforms both controls and collapses VQ. **WHY:** this invalidates the positive representation claim. **HOW:** finish the frozen mode sweep, inspect per-term gradient conflict, then remove/redesign terms under a new preregistered protocol; do not tune on held-out test. **WHERE:** `ripii/models/ripii.py`, `ripii/utils/loss_balancer.py`, `scripts/run_post_correction.py`, `research/protocols/`. **VERIFY:** ≥5 development seeds first pass predeclared reconstruction/probe/non-collapse gates, followed by a fresh locked confirmatory seed set.
- [ ] **Repair the overnight study runner.** **WHAT:** calibration accepted zero per-step time, first cell failed, error context was suppressed, and no resume exists. **WHY:** the planned evidence cannot be generated reliably. **HOW:** repeat warmed timing points, fit a robust upper-bound model, reject inconsistent data, capture stdout/stderr per cell, atomically mark completion, and resume from verified cells. **WHERE:** `scripts/run_post_correction.py`, `scripts/start_post_correction_overnight.sh`, `scripts/post_correction_status.sh`. **VERIFY:** fault-injection integration test kills one cell, status reports the exact error, restart skips verified cells, and a miniature matrix completes twice identically.
- [ ] **Resolve legal/submission blockers.** **WHAT:** license, ownership, authorship, affiliations, conflicts, funding, and citation metadata are absent. **WHY:** reuse and submission are not legally or procedurally complete. **HOW:** obtain owner decisions and add standard files/sections. **WHERE:** repository root, `pyproject.toml`, `paper/`. **VERIFY:** automated release check and human owner sign-off.
- [ ] **Freeze a clean evidence revision.** **WHAT:** the audit occurred on a heavily dirty tree with stale historical reports. **WHY:** reviewers cannot identify a single authoritative state. **HOW:** separate intended changes, rerun gates, tag the exact commit, and emit source/data/artifact hashes. **WHERE:** Git history, `EVIDENCE_LEDGER.md`, release bundle. **VERIFY:** clean checkout reproduces all release checks and every retained digest.

### P1 — required for credible research

- [ ] **Run adequately powered external confirmation.** **WHAT:** current n=3/5 studies cannot produce conventional exact significance. **WHY:** effect direction and uncertainty remain weakly estimated. **HOW:** lock the primary metric/model contrast and run the preregistered external protocol with at least the planned seed count, revising upward from external-pilot variance only. **WHERE:** `research/protocols/confirmatory_external_v1.md`, study runners. **VERIFY:** report all seeds, paired effect/CI, exact/randomization test, multiplicity rule, exclusions, and deviations.
- [ ] **Add strong contemporary baselines.** **WHAT:** current controls omit equivariant, noise-trained, constrained, and established multiscale simulators. **WHY:** novelty and competitiveness cannot be judged against the field. **HOW:** implement or use authors’ official code with matched data/tuning; include EGNN, GNS, constraint-based GNS, MGN/BSMS, and energy/Lagrangian approaches as applicable. **WHERE:** `ripii/world/models.py`, new adapters/configs/protocol. **VERIFY:** unit parity tests, parameter/FLOP/time tables, convergence curves, and independently checked metrics.
- [ ] **Make comparisons budget-fair.** **WHAT:** updates/parameters are matched, convergence/compute/search are not. **WHY:** rankings may reflect optimization allocation. **HOW:** predefine equal wall-clock/FLOP and convergence criteria plus equal hyperparameter budgets. **WHERE:** `world/experiment.py`, profiling scripts, protocols. **VERIFY:** publish learning curves, peak memory, FLOPs, time, chosen hyperparameters, and all trials.
- [ ] **Archive complete rerunnable evidence.** **WHAT:** historical checkpoints, raw NRI arrays, split indices, and logs are absent. **WHY:** hash verification alone cannot reproduce inference or paper numbers. **HOW:** create a versioned artifact bundle/DOI with raw or fetch-verified data, prepared-data hashes, checkpoints, logs, configs, environment lock, and commands. **WHERE:** release tooling, `research/results/`, data registry. **VERIFY:** offline clean-machine replay regenerates evaluation JSON and paper tables within declared tolerance.
- [ ] **Use a genuine external benchmark.** **WHAT:** NRI is a second synthetic generator with padded semantics. **WHY:** external validity remains untested. **HOW:** add a licensed community benchmark and noisy/empirical dataset; document domain mapping. **WHERE:** `world/external_data.py`, dataset cards, new protocol. **VERIFY:** schema/unit tests, checksums, train-only normalization, no-overlap audit, and locked test evaluation.
- [ ] **Align physical architecture and claims.** **WHAT:** absolute-coordinate MLPs lack Euclidean equivariance and use arbitrary residual scales. **WHY:** physical generalization claims are unsupported. **HOW:** add relative/equivariant models, train-set standardization, scale ablations, and a no-position-bypass control. **WHERE:** `world/models.py`, `experiment.py`, configs. **VERIFY:** numerical symmetry tests plus OOD unit/rotation/translation results.
- [ ] **Correct NRI semantics and reporting.** **WHAT:** constant radius/mass/zero action are not observations and edge labels are unused. **WHY:** readers may infer physical or relational information the model never receives. **HOW:** mark padded channels explicitly, mask/remove meaningless metrics, test edge-aware and edge-blind variants, and report the exact task. **WHERE:** `world/nri_data.py`, NRI protocol/manuscript. **VERIFY:** adapter tests assert provenance masks; paper table lists real versus constructed fields.
- [ ] **Build a real paper artifact from evidence.** **WHAT:** manuscript/build are scaffolds. **WHY:** no reviewable, traceable submission exists. **HOW:** produce LaTeX/Quarto, numbered equations, figures/tables generated from immutable JSON, methods detail, limitations, and supplement. **WHERE:** `paper/`, `scripts/build_paper.sh`. **VERIFY:** clean CI build yields PDF and fails if cited result hashes drift.

### P2 — major quality improvements

- [ ] **Redesign objective scalarization.** **WHAT:** adaptive totals are incomparable and assumptions are unproved. **WHY:** optimization can reward log-variance movement rather than scientific performance. **HOW:** choose fixed normalized weights or a justified multiobjective method; log raw losses, weights, cosine conflicts, and gradient norms. **WHERE:** `loss_balancer.py`, `objective_diagnostics.py`, training logs. **VERIFY:** toy analytic tests, finite-difference gradients, and stable rankings under rescaling.
- [ ] **Extend physical evaluation.** **WHAT:** energy/contact/conditional conservation and distributional metrics are missing. **WHY:** RMSE alone can favor physically implausible rollouts. **HOW:** add work-energy residuals, collision penetration/impulse error, closed-system momentum, spectral/distributional trajectory metrics, and failure-stratified results. **WHERE:** `world/experiment.py`, `failure_analysis.py`. **VERIFY:** analytic simulator cases and independent reference calculations.
- [ ] **Match rollout training to deployment.** **WHAT:** four-step training is evaluated at 32 steps. **WHY:** error accumulation is not directly controlled. **HOW:** ablate horizon curricula, state noise, scheduled rollout, and stability regularization. **WHERE:** world config/training loop. **VERIFY:** error-versus-horizon curves to substantially beyond 32 steps with divergence rate.
- [ ] **Implement sparse interactions and scaling tests.** **WHAT:** local models still allocate dense pairs. **WHY:** claimed hierarchy/scaling advantages cannot emerge credibly. **HOW:** neighbor lists or sparse edges with segment reductions; benchmark N from 8 to the memory limit. **WHERE:** `world/models.py`, profiling scripts. **VERIFY:** output parity on small cases and empirical time/memory scaling slopes.
- [ ] **Improve VQ robustness.** **WHAT:** dead-code collapse persists. **WHY:** the discrete hierarchy is functionally absent. **HOW:** compare EMA updates, code resets, commitment schedules, diversity regularization, smaller codebooks, and FSQ-style quantization.[^7] **WHERE:** `models/quantizer.py`, configs, qualification script. **VERIFY:** preregistered minimum usage/perplexity, stability across seeds, and no reconstruction/probe regression.
- [ ] **Strengthen CI reproducibility.** **WHAT:** CI does not enforce the lock or types/coverage/performance. **WHY:** dependency drift and silent untested paths remain. **HOW:** frozen `uv` install, type checker, coverage threshold, packaging smoke, deterministic mini end-to-end and benchmark budget. **WHERE:** `.github/workflows/`, `pyproject.toml`. **VERIFY:** clean CI matrix passes and intentional drift/type/perf regressions fail.
- [ ] **Add dataset/model cards.** **WHAT:** provenance and intended-use metadata are scattered. **WHY:** reviewers cannot audit units, splits, limitations, or licenses efficiently. **HOW:** standardized cards with hashes and field-level provenance. **WHERE:** `docs/data/`, `docs/models/`. **VERIFY:** release checklist requires all fields and resolves every artifact hash.

### P3 — polish and optimization

- [ ] **Consolidate truth documents.** **WHAT:** multiple audits/checklists contain stale counts/status. **WHY:** ambiguity wastes reviewer time. **HOW:** create a dated index with `current`, `superseded`, and `historical` labels. **WHERE:** `audit/README.md`, root README links. **VERIFY:** link checker and one authoritative current-status pointer.
- [ ] **Catalogue failed and development runs.** **WHAT:** incomplete run directories lack a uniform registry. **WHY:** negative operational evidence can be lost or mistaken for final results. **HOW:** machine-readable run index with status, cause, source hash, and retention policy. **WHERE:** `runs/index.json`, evidence ledger. **VERIFY:** status tool accounts for every study directory.
- [ ] **Improve release ergonomics.** **WHAT:** no SBOM, DOI, citation file, changelog-linked tag, or command transcript bundle. **WHY:** external reuse is harder. **HOW:** add release workflow and archival metadata. **WHERE:** root/release CI. **VERIFY:** install and reproduce from release artifact, not working checkout.
- [ ] **Document performance and platform support.** **WHAT:** no tested device/OS matrix or deterministic tolerance table. **WHY:** users cannot distinguish numerical drift from failure. **HOW:** record CPU/CUDA/MPS support, expected runtime/memory, and tolerance policy. **WHERE:** `docs/REPRODUCIBILITY.md`, CI. **VERIFY:** scheduled platform smoke jobs and published results.

## Final reviewer assessment

RIPII is **not vaporware**: the main mechanisms, controls, experiment runners, metrics, and integrity tooling are implemented and tested. It is also **not a validated research method**: the experiments consistently fail to show the proposed hierarchy/full objective beating simpler alternatives, the strongest baselines and external data are missing, and the evidence archive cannot reconstruct every paper number from raw inputs.

The highest-value path is not to add more mechanisms. It is to make the negative result airtight: complete the objective localization study, add the strongest baselines, run a powered external protocol, archive every dependency and checkpoint, and publish a reproducible negative-results paper about when learned hierarchy and multi-loss structure fail. If later changes reverse the result, they must be evaluated under a new frozen protocol and untouched confirmatory data—not retrofitted into the present development evidence.

## Sources

[^1]: Battaglia et al., “Interaction Networks for Learning about Objects, Relations and Physics,” NeurIPS 2016. https://proceedings.neurips.cc/paper_files/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html
[^2]: Sanchez-Gonzalez et al., “Learning to Simulate Complex Physics with Graph Networks,” ICML 2020. https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
[^3]: Kipf et al., “Neural Relational Inference for Interacting Systems,” ICML 2018. https://proceedings.mlr.press/v80/kipf18a.html
[^4]: Satorras et al., “E(n) Equivariant Graph Neural Networks,” ICML 2021. https://proceedings.mlr.press/v139/satorras21a.html
[^5]: Fortunato et al., “MultiScale MeshGraphNets,” arXiv:2210.00612. https://arxiv.org/abs/2210.00612
[^6]: Cao et al., “Accelerating Mesh-based Simulations with Bi-Stride Multi-Scale Graph Neural Networks,” ICML 2023. https://proceedings.mlr.press/v202/cao23a.html
[^7]: Mentzer et al., “Finite Scalar Quantization: VQ-VAE Made Simple,” arXiv:2309.15505. https://arxiv.org/abs/2309.15505
[^8]: Toshev et al., “LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite,” NeurIPS 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/ccac3b120c7dc86d45f56830732b62be-Abstract-Datasets_and_Benchmarks.html
[^9]: Ohana et al., “The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning,” NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f9a5acd91ac76569f2fe291b1f4772b-Abstract-Datasets_and_Benchmarks_Track.html
[^10]: Rubanova et al., “Constraint-based Graph Network Simulator,” ICML 2022. https://proceedings.mlr.press/v162/rubanova22a.html
[^11]: Cranmer et al., “Lagrangian Neural Networks,” arXiv:2003.04630. https://arxiv.org/abs/2003.04630
[^12]: Xu et al., “Equivariant Graph Neural Operator for Modeling 3D Dynamics,” ICML 2024. https://proceedings.mlr.press/v235/xu24j.html
