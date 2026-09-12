# NRI external-simulator development protocol v1

Status: **frozen development protocol**. Frozen before any RIPII model was trained or
evaluated on the NRI test trajectories. This is not a confirmatory protocol and must
not be relabeled after results are observed.

## Question and hypotheses

On two independently implemented interacting-object simulators, does RIPII's learned
multiscale grouping improve rollout position RMSE over a parameter-matched global-pool
control?

- **H0:** multiscale grouping provides no practically meaningful advantage over global
  pooling.
- **H1 development gate:** multiscale position RMSE is at least 5% lower than global
  pooling for every paired seed on IID test trajectories from both NRI domains.

The independent variable is model interaction mechanism. The primary dependent variable
is full-rollout position RMSE. Velocity RMSE and horizon-specific position RMSE are
secondary diagnostics. Initialization/minibatch seed is the paired unit; frames, objects,
and scenes are not replicates.

## Data fixed before evaluation

- Source: official `ethanfetaya/NRI` repository.
- Commit: `e63fcb0144bca60eb1cffed9a94489de928d6c23`.
- License recorded from upstream: MIT.
- Domains: `SpringSim(n_balls=5)` and `ChargedParticlesSim(n_balls=5)`.
- Base seed: 41003. Domain offsets: springs 0, charged 10000. Split offsets:
  train 0, validation 100000, test 200000.
- Train: 128 trajectories × 17 observations. Validation: 32 × 17. Test: 32 × 33.
- Upstream integration time step: 0.001; sample frequency: 50; model observation time
  step: 0.05.
- Positions and velocities are divided by the simulator's fixed box size 5. No fitted
  preprocessing statistic is used. Edge labels are retained for provenance but never
  provided to any model.
- Radius 0.04 and mass 1 are fixed, non-predictive object properties required by the
  common RIPII state contract. Actions are zero because NRI supplies autonomous systems.

The generated archive manifest must verify before training. Springs and charged are
two external simulator domains, not two real-world datasets.

## Models, selection, and budgets

Analytic controls: persistence, constant velocity, and force kinematics. Neural controls:
MLP, flat local graph, Transformer, global pool, and multiscale. Neural widths are selected
before training as the closest multiple-of-four width to the multiscale parameter count;
any parameter mismatch above 5% aborts the run.

- Seeds: 1009, 1013, 1019.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001.
- Updates: 100 per cell.
- Batch size: 32; four-step rollout objective; validation every 20 updates.
- Selection: minimum validation position RMSE + 0.25 × velocity RMSE.
- No early stopping and no test-driven retry. Any missing, failed, or non-finite cell
  invalidates the complete study.

All neural models receive identical states, masks, zero actions, trajectories, update
counts, validation rule, and paired sampler seeds. This is parameter- and update-matched,
not compute-matched; machine-local timing is descriptive only.

## Evaluation and analysis

Each trained model is evaluated after validation selection on:

1. IID held-out trajectories from its training dynamics law;
2. the other NRI law without adaptation (`law_shift`).

Raw seed metrics are retained. Dataset-specific paired summaries and a fixed-case average
over the two domains are reported with mean/median/sample SD, deterministic paired
bootstrap 95% intervals, and exact two-sided sign-flip tests. With three development
seeds, inferential power is deliberately inadequate.

The gate passes only if multiscale beats global pool by at least 5% on IID position RMSE
for all six domain-seed pairs. Passing authorizes a powered prospective protocol; it is
not confirmation. Failing rejects hierarchy-specific advancement on these settings.

## Evidence boundary

Classification: **external-simulator development evidence**. The simulators are external
code, but trajectories are locally generated after method development. This study cannot
support real-world generalization, population-level significance, state of the art,
compute efficiency, or a confirmatory claim. A powered protocol requires at least 18
paired seeds subject to upward revision from these pilot variances and must add a frozen
compute tolerance before test evaluation.
