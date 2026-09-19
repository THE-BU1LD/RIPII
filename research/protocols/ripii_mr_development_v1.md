# RIPII-MR Development Protocol v1

Status: **prospective development protocol; no result exists at freeze time**.

## Question

Does a geometry-aware, E(2)-equivariant, internally conservative and adaptively routed
multiresolution simulator improve long-range/OOD rollout accuracy over matched learned
controls in the coupled soft-disc generator?

## Candidate and controls

- Candidate: `ripii_mr`, continuous path only.
- Controls: `equivariant`, `graph`, `global_pool`, and legacy `multiscale`.
- Analytic references: persistence, constant velocity, and force kinematics.
- Capacity matching: nearest hidden width to the existing multiscale parameter budget,
  with a maximum 5% parameter-count difference. Compute and wall time are reported but
  are not guaranteed equal in this development study.

## Frozen default matrix

- Data seeds: 3101, 3203, 3307.
- Model/minibatch seeds: 131, 137, 139, 149, 151.
- Training: 1,200 updates, 384 scenes, horizon 24, rollout curriculum to 8 steps over
  600 updates, state noise 0.002, batch size 32.
- Validation: 96 held-out scenes every 100 updates; select minimum position RMSE plus
  0.25 times velocity RMSE.
- Test: 96 scenes per IID/OOD regime, horizon 64.
- Physics: long-range coupling 1.0 plus the declared contact/wall/drag simulator.

## Primary endpoint and advancement

For each paired data/model seed, average position RMSE across the `more_objects`,
`composition`, and `fast` splits. RIPII-MR must improve that mean by at least 5% over
**every** included learned control in **every** paired cell, while its IID position RMSE
regresses by no more than 5% in every comparison. Any missing, nonfinite or failed cell
is a failure. In plain terms, every paired cell must pass both thresholds. Passing yields
only `advance_development`; otherwise `no_advance`.

## Evidence boundary

This is synthetic development evidence, not confirmation. The three data seeds are
independent generator draws; model seeds nested inside a data seed are not independent
datasets. External public benchmarks, compute-matched tuning, independent reproduction,
and a separately frozen untouched test remain required for a publication claim.

## Execution

```bash
./scripts/start_ripii_mr_overnight.sh runs/ripii_mr_development_v1
.venv/bin/python scripts/run_ripii_mr_matrix.py \
  --output runs/ripii_mr_development_v1 --verify-only
```

The runner writes its machine-readable protocol before training, snapshots the source in
each child benchmark, resumes only verified complete child data-seed runs, retains
failures, and emits a content-hashed outer manifest.
