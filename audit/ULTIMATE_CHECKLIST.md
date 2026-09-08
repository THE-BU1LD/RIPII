# RIPII ultimate execution checklist

Updated: 2026-09-09. This is the single plain-language backlog after the deep audit.
Checked items are already real and verified. Unchecked items still need work. Items
marked `OWNER_REQUIRED` or `EXTERNAL_EXECUTION_REQUIRED` cannot honestly be completed
from the current repository alone.

## 0. What is fake, pseudocode, placeholder, or scaffolded?

- [x] **P0 — Executable pseudocode audit complete.** No pseudocode, `pass`, stub,
  `NotImplementedError`, fake prediction, hand-authored metric generator, or mock-data
  substitution was found on a canonical execution path. Evidence: source inspection,
  repository-wide pattern scan, and 97 passing tests.
- [x] **P0 — “Scaffold” references classified.** The node/fusion scaffold mentioned in
  pilot v2 is real executable model structure, not an unfinished code scaffold. It is
  called a scaffold only because the `no_structured` control retains it.
- [x] **P2 — Placeholder config classified.** `runs/pilot_v2_placeholder` in
  `configs/pilot_v2.yaml` is an intentionally overridden output-path sentinel. It does
  not generate fake data or results. The frozen config must not be rewritten.
- [x] **P0 — Smoke profiles classified.** `configs/smoke.yaml` is plumbing-only;
  `configs/mechanism_smoke.yaml` executes the mechanisms. Neither is research evidence.
- [x] **P0 — Historical defects classified.** Pilot v1 is not repaired or promoted. Its
  loss-balancer defect is preserved as invalidating evidence for cross-mode total-loss
  ranking.

## 1. Bad or scientifically risky implementations

- [x] **P0 — Zero-weight adaptive-loss bug fixed prospectively.** Disabled losses no
  longer contribute learned offsets or gradients. Verify: loss-balancer regression tests.
- [x] **P0 — Evaluation weighting fixed.** Short final batches are weighted by sample
  count. Verify: `test_epoch_metrics_weight_samples_including_short_batch`.
- [x] **P0 — Data leakage controls implemented.** Train/validation/test indices and
  generated split domains are disjoint. Verify: split tests.
- [x] **P0 — Best-checkpoint behavior implemented.** World models select on validation,
  never test. Exact resume retains RNG, optimizer, and best score.
- [x] **P0 — Invalid numerics fail closed.** NaN/Inf, empty scenes, corrupt checkpoint
  schemas, incompatible resumes, invalid benchmark grids, missing artifacts, and stale
  hashes raise errors.
- [x] **P1 — Weak legacy baseline corrected.** A genuinely plain autoencoder exists and
  is parameter matched without unused padding.
- [ ] **P1 — Legacy objective remains over-complex.** Thirteen interacting loss terms
  are weakly identifiable and may fight through global clipping. Required: run a frozen
  objective study comparing reconstruction+KL against individually added terms, logging
  raw magnitudes, per-loss gradient norms, and cosine similarities. Accept improvement
  only if held-out behavior changes consistently.
- [x] **P1 — Unconstrained grouping challenged.** Failure localization shows the deficit
  is broad rather than contact-specific. A prospective long-range force intervention
  also fails to create a hierarchy-specific advantage over global pooling. Current
  evidence does not justify adding a locality penalty.
- [ ] **P1 — Equal updates are not equal compute.** Instrumentation is now complete:
  `profile_world.py` records warmup, repeated synchronized trials, bootstrap intervals,
  throughput, checkpoint size, environment, accelerator memory where available, and a
  labeled PyTorch-recognized FLOP lower bound. Remaining: freeze and run a three-model
  comparison within a predeclared 5% compute tolerance.
- [ ] **P2 — Dense interaction scaling is poor.** Local graph code constructs all
  object pairs before applying the distance mask: $O(N^2h)$. Required only for larger
  object counts: abstract neighbor construction and benchmark sparse radius graphs.
- [x] **P2 — Smoke runner process isolation.** Training, evaluation, diagnostics, and
  reporting now run as subprocesses so PyTorch thread pools and Matplotlib backends do
  not leak process-global state across stages. Verify with warning-free canonical smoke.

## 2. Improve what already exists

- [x] **P1 — Add prospective power design.** The self-checksummed provisional plan uses only v3
  development-seed relative differences, fixes a 5% minimum detectable effect, two-sided
  alpha 0.05, and 80% target power, and recommends 18 paired seeds by a documented normal
  approximation. External pilot variance may increase—not decrease—the frozen count.
- [x] **P1 — Add dataset registry abstraction.** `DatasetAdapter`/`DatasetSpec` records
  identity, version, license metadata, units, masks, split policy, preprocessing, tensor
  bytes, and deterministic content SHA-256. The simulator is the built-in adapter;
  external adapters still require actual owner-selected datasets.
- [x] **P1 — Add protocol schema validation.** `ExperimentProtocol` is immutable,
  canonical, hashable, versioned, and validates model grids, datasets, seeds, budgets,
  source hashes, selection, and decision rules before creating new output directories.
- [x] **P1 — Add run-status state machine.** New world/coupling runs bind
  `planned -> running -> complete` into their manifests or retain a typed `failed` event.
  Self-checksummed failure-analysis artifacts record the separate post-result analyzed state;
  partial grids never receive a decision or valid manifest.
- [ ] **P1 — Archive full checkpoints externally.** The compact self-checksummed v3 capsule proves
  summaries but cannot re-evaluate models. Store full manifested runs in durable,
  content-addressed storage and record retrieval instructions.
- [x] **P2 — Add contact-conditioned failure analysis.** Errors are reported separately
  for contact, near-contact, free-flight, wall-contact, and forced transitions. The five
  seeds—not timesteps—remain paired units. Evidence: `world_v3_failure_analysis.json`.
- [ ] **P2 — Complete calibration and stability diagnostics.** Rollout growth, corrected
  arena escapes, scene quantiles, worst scene, property drift, momentum error, and maximum
  state magnitudes are now reported. Remaining: predeclare calibration thresholds and add
  Jacobian sensitivity only if a future stability hypothesis requires it.
- [x] **P2 — Add robust timing harness.** `profile_world.py` includes warmup, repeated
  trials, synchronization, device details, bootstrap intervals, accelerator peak memory
  where supported, throughput, parameters, checkpoint size, and an explicit FLOP boundary.
- [x] **P2 — Separate legacy and world-model namespaces more clearly.** `docs/ARCHITECTURE.md`
  names their distinct questions and paths while retaining shared statistics utilities.
- [x] **P3 — Reduce generated working-tree clutter.** `clean_generated.py` defaults to a
  dry run and can remove only explicit build/test caches. It deliberately excludes all
  `runs/` evidence.

## 3. Add missing scientific evidence

- [ ] **P0 — Choose and license external datasets.** At least two public object-dynamics
  datasets or independent simulators are required. Record source, version, checksum,
  license, units, exclusions, and preprocessing. `OWNER_REQUIRED` for acceptable licenses.
- [ ] **P0 — Freeze the external confirmatory protocol.** Complete dataset choices,
  hardware, seeds, budgets, power, primary metric, effect threshold, model selection,
  stopping, statistics, failure rules, and external timestamp before examining test
  outcomes.
- [ ] **P1 — Run strong external baselines.** Persistence, constant velocity, force
  kinematics where compatible, MLP, flat graph, Transformer, global pool, and multiscale.
  Match information access, preprocessing, selection, and compute.
- [ ] **P1 — Run the justified paired seed count per dataset.** The current provisional
  lower bound is 18 pairs, subject to upward revision from external pilot variance.
  Preserve every failure and seed. `EXTERNAL_EXECUTION_REQUIRED`.
- [ ] **P1 — Run genuine external OOD.** Separate new scenes from new object counts,
  physical parameters, dynamics laws, temporal regimes, missing observations, sensor
  noise, and cross-dataset transfer.
- [ ] **P1 — Run objective ablations.** Reconstruction-only/simple objective, each added
  regularizer, frozen/randomized grouping, uniform grouping, global pooling, and removed
  hierarchy. Test that flags alter executed computation.
- [ ] **P1 — Run capacity and compute controls.** Parameter-matched, compute-matched,
  depth-matched, receptive-field-matched, and training-budget sensitivity comparisons.
- [ ] **P2 — Complete scaling studies.** Global coupling and larger-object evaluation
  now have an initial controlled study; still vary training data, rollout horizon,
  width, groups, and compute. Report raw seed curves and failure rates.
- [ ] **P2 — Run lower-data and corrupted-input studies.** Predeclare relevant noise,
  missing-object, and action-corruption levels.
- [ ] **P2 — Obtain independent reproduction.** A researcher not involved in development
  must reproduce installation, smoke, one external run, analysis, and artifact checking.

## 4. Statistics and claims

- [x] **P1 — Seed-level statistics implemented.** Raw differences, mean, median, sample
  SD, deterministic paired bootstrap summaries, exact sign-flip tests, and Holm adjustment
  are available and tested.
- [x] **P0 — Correct experimental unit documented.** Seeds are units in current paired
  comparisons; objects, frames, and timesteps are not independent replicates.
- [ ] **P1 — Add hierarchical multi-dataset analysis.** Pair seeds within dataset and
  aggregate datasets explicitly. Predeclare whether datasets are fixed cases or sampled
  from a target population.
- [x] **P1 — Add failure-inclusive analysis.** Current protocols use the strict rule that
  any failed cell fails the whole run and produces no decision or valid manifest; failed
  cells cannot be silently omitted. A future population analysis may predeclare a bounded
  penalty instead.
- [x] **P1 — Add multiplicity families.** World analysis explicitly defines the four
  position-RMSE splits as one Holm-adjusted family; the primary advancement rule remains
  separate and predeclared.
- [x] **P2 — Add practical-equivalence analysis.** Paired relative errors, a ±5% margin,
  fixed-seed bootstrap interval, all-seed result, and non-population claim boundary are
  now generated prospectively.
- [x] **P0 — Unsupported wording removed.** No “significant,” “state of the art,” novel,
  efficient, invariant, or external-generalization claim is currently authorized.

## 5. Abstractions worth adding

- [x] **P1 — `DatasetSpec`/loader protocol.** One interface for tensors, units, masks,
  trajectory IDs, split IDs, hashes, and provenance.
- [x] **P1 — `ExperimentProtocol` model.** Validated, immutable after freeze, hashable,
  and serializable; command generation must come from this object.
- [ ] **P1 — `RunRecord` model.** World runs now combine status, protocol, dataset,
  environment, seed, budget, checkpoint, metrics, and manifest provenance. Remaining:
  migrate the frozen legacy path to a common read-only record without rewriting history.
- [ ] **P1 — Evaluator registry.** Shared fail-closed metric aggregation with explicit
  units and experimental unit; avoid architecture-specific hand assembly.
- [ ] **P2 — Bottleneck interface.** Continuous, FSQ, and VQ should expose a common
  `(output, auxiliary_loss, diagnostics)` contract instead of mutable side channels.
- [ ] **P2 — Neighbor/interaction interface.** Separate edge construction from message
  updates so dense, radius, k-NN, and supplied graphs can be controlled fairly.
- [ ] **P2 — Artifact store interface.** Local and remote content-addressed storage with
  the same manifest verification contract.
- [ ] **P3 — Reporter interface.** Generate JSON first, then Markdown/tables/figures from
  retained JSON; never allow manuscript-only numbers.

## 6. Additions that are justified only after diagnostics

- [x] **P2 — Spatially constrained grouping decision.** Not implemented after diagnostics:
  failure is not contact-localized, and simple global pooling matches or beats hierarchy
  under explicit long-range forces. Extra grouping constraints lack current justification.
- [ ] **P2 — Contact-event auxiliary loss.** Add only if contact timesteps dominate error.
  Supply identical supervision to controls. This changes information access and must not
  be presented as proof of the original hierarchy.
- [ ] **P2 — Conditional/dynamic coarse computation.** Add only for an efficiency
  hypothesis with measured FLOPs and latency. Current hierarchy is not adaptive compute.
- [ ] **P3 — Learned perception.** Treat pixels-to-objects as a separate hypothesis and
  benchmark perception error independently; do not hide it inside dynamics results.
- [ ] **P3 — Alternative quantization.** FSQ or collapse-resistant VQ is justified only
  for a compression/discrete-representation question. Current evidence supports removing
  VQ from the main method.

## 7. Repository, release, and paper

- [x] **P1 — Canonical commands exist.** Preflight, tests, mechanism smoke, development
  experiment, analysis, manuscript check, and artifact verification are scripted.
- [x] **P1 — Required audit/research documents exist.** Repository map, mathematical
  specification, hypotheses, novelty audit, three reviews, truth statement, evidence
  ledger, final audit, report, protocols, and manuscript draft are present.
- [ ] **P0 — Add owner-approved license.** Do not infer one. `OWNER_REQUIRED`.
- [ ] **P0 — Add authorship, affiliations, contribution statement, and conflicts.**
  `OWNER_REQUIRED`.
- [ ] **P1 — Choose the honest paper contribution.** Either (a) a powered multi-dataset
  negative result about learned hierarchy, or (b) a genuinely new, supported mechanism.
  Current evidence does not support a positive method paper.
- [ ] **P1 — Generate every paper table/figure from retained artifacts.** Add claim IDs
  linking manuscript statements to raw JSON and protocol hashes.
- [ ] **P1 — Build venue-formatted PDF and supplementary artifact.** Only after evidence,
  ownership, license, and venue are fixed.
- [ ] **P1 — Run anonymous clean-machine artifact reproduction.** Verify install, tests,
  smoke, analysis, figures, paper, and hashes without machine-local state.
- [ ] **P1 — Archive release artifacts immutably.** Source tag, protocol digests, data
  metadata, raw metrics/predictions, checkpoints, logs, paper, and artifact manifest.

## 8. Final go/no-go gates

- [x] **Engineering gate:** core code is real, fail-closed, tested, packaged, and locally
  reproducible.
- [ ] **Experiment-ready gate:** external dataset adapters, complete frozen protocol,
  compute measurement, and storage must exist.
- [ ] **Evidence gate:** powered external runs, robustness, ablations, statistics, and
  independent reproduction must complete without dropped failures.
- [ ] **Submission gate:** novelty/contribution is defensible, all claims trace to raw
  artifacts, owner metadata/license exist, and three adversarial reviews have no fatal
  concern.

Current status: **EVIDENCE_PARTIAL**. The immediate sequence is: owner dataset selection
and license review → implement those adapters → freeze a powered compute-matched external
protocol → cheap external smoke → powered paired execution → independent reproduction →
paper decision.
