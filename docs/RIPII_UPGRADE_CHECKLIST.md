# RIPII upgrade checklist

This is a whole-project upgrade plan. It keeps the existing negative evidence intact:
the objective, quantizer, and multiscale mechanisms should not be "improved" by
post-hoc tuning until a new falsifiable hypothesis and protocol are written.

## P0 — scientific and training foundation

- [ ] Select one primary RIPII hypothesis (rather than extending both the legacy
  latent stack and world-model hierarchy in parallel) and write its predicted
  failure regime, comparison, metric, effect threshold, compute budget, and stop rule.
- [ ] Freeze a convergence- and compute-matched protocol with fresh paired seeds;
  record wall time, FLOP lower bounds, peak memory, and parameter counts per cell.
- [ ] Make baseline selection comprehensive: simple reconstruction+KL, plain AE,
  MLP, graph, Transformer, global pool, analytic references, and the best supported
  non-RIPII control for the target task.
- [ ] Separate exploratory sweeps from confirmation: tune only in a development
  split, lock choices, then rerun a fresh confirmation suite.
- [ ] Investigate the observed objective-gradient conflict and codebook collapse
  with a predeclared intervention study before retaining VQ or auxiliary losses.
- [ ] Add principled early-stop/convergence criteria and report learning curves,
  seed-level outcomes, uncertainty intervals, and all failed cells.

## P0 — overnight reliability

- [ ] Reproduce the failing `base` / seed `1009` training child command from the
  quarantined attempt with stdout and stderr visible.
- [ ] Identify and fix the underlying training failure; do not treat the benchmark
  wrapper's nonzero exit as the root cause.
- [ ] Add a regression test for the exact failure once its root cause is known.
- [ ] Run the repaired `base` / seed `1009` cell at 600 steps, then verify that it
  produces `final.pt`, `eval.json`, retained artifacts, and a valid manifest.
- [ ] Run a representative multi-seed, multi-mode smoke sweep using the frozen
  protocol options before scheduling all 180 cells.

## P1 — architecture, data, and inference

- [ ] Publish stable, versioned contracts for legacy structured-latent and world
  paths, including tensor shapes, units, masks, action semantics, checkpoint schema,
  and compatibility policy.
- [ ] Decide whether the legacy path is frozen/archived or actively supported;
  avoid adding mechanisms to both paths without a shared hypothesis.
- [ ] Add real-observation or clearly scoped external-data adapters with documented
  licenses, dataset cards, split fingerprints, units, normalization, and leakage tests.
- [ ] Expand OOD evaluation along independently controlled axes: object count,
  dynamics law, action distribution, observation noise, rollout horizon, and missingness.
- [ ] Define an inference-serving contract with batch behavior, error handling,
  deterministic preprocessing, model/version metadata, and reproducible rollout APIs.
- [ ] Add numerical stability checks for long rollouts, invalid masks, extreme
  transforms/actions, zero-object cases, and non-finite model outputs.

## P1 — engineering and evaluation

- [ ] Split benchmark orchestration from training so each cell has structured
  stdout/stderr, exit reason, resource telemetry, and independently rerunnable commands.
- [x] Preserve child stdout and stderr when a benchmark subprocess fails.
- [ ] Add regression tests for failed-cell reproduction, checkpoint resume, manifest
  drift, protocol immutability, and all public CLI/API contracts.
- [ ] Run the full test suite across every supported Python version in CI and add
  deterministic CPU reference tests for numerical regressions.
- [ ] Add property/fuzz tests for configuration parsing, manifests, NPZ inputs,
  checkpoint validation, and dataset split/fingerprint handling.
- [ ] Standardize experiment artifacts: machine-readable metrics, environment lock,
  source revision, data fingerprint, seed policy, command, and retained logs.
- [ ] Profile training and inference systematically, then prioritize actual bottlenecks
  instead of optimizing parameter count alone.

## P1 — make launch and recovery dependable

- [ ] Add a launcher option or environment variable for an explicit output
  directory, and test that the supplied path is passed through unchanged.
- [ ] Start the launcher from a persistent host process (Terminal, `launchd`, or
  an equivalent supervisor), rather than relying on a sandbox-detached shell.
- [ ] Record the supervisor PID, command, working directory, start time, and
  environment in `launch.json`.
- [ ] Add a liveness check that distinguishes `running`, `completed`, `failed`,
  and a stale PID file.
- [ ] Flush or line-buffer the overnight log so progress and failures are visible
  while a mode is running.
- [ ] On startup, reject a failed/stale output directory with an actionable
  message; require a new output directory or an explicit recovery action.
- [ ] Keep failed attempts immutable and launch the repaired matrix into a fresh,
  timestamped output directory.

## P2 — security, packaging, and release

- [ ] Choose and add an owner-approved license, authorship/contributor policy,
  citation metadata, changelog discipline, and a supported-version policy.
- [ ] Publish a minimal-install path and test a clean-wheel install, CLI invocation,
  and inference smoke test in CI.
- [ ] Produce versioned checkpoint/model cards with intended use, limitations,
  known negative results, trusted-loading requirements, and provenance hashes.
- [ ] Enable repository secret scanning and push protection; add a release-time
  secret/path scan and dependency/SBOM artifact.
- [ ] Define a vulnerability-response SLA and responsible-disclosure owner.
- [ ] Write user-facing tutorials for training, evaluation, rollout inference,
  external datasets, reproducibility, and interpreting `no_advance` outcomes.

## P2 — validate a full study run

- [ ] Recalibrate after the training fix; retain the calibration measurements and
  confirm the 180-cell estimate is below the 24-hour cap.
- [ ] Run preflight and verify the frozen source/config/protocol hashes are
  recorded before any cell starts.
- [ ] Verify one completed mode manifest and one retained-run artifact set using
  the production verification path.
- [ ] Confirm the runner skips only complete, manifest-verified modes and
  quarantines incomplete groups without overwriting them.
- [ ] Run the focused runner and benchmark tests, then the full test suite and
  `git diff --check`.

## Go / no-go

Launch a full overnight matrix only when the overnight P0 items and relevant P1
engineering gates are checked. Call RIPII research-ready only after the scientific P0
items, external-data evaluation, and independent reproduction are complete. Do not
turn a development win into a method or publication claim without those gates.
