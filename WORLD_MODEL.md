# RIPII object-state world model

This is a complete, local train/evaluate/intervene workflow. It predicts object
positions and velocities from current object states and applied forces. It does not
infer objects from pixels and it does not establish a novel or superior method.
The earlier vector-reconstruction architecture and its immutable failed pilots remain
available separately.

## Run it

Install the repository using `uv sync --locked --extra dev` or
`python -m pip install -e '.[dev]'`. The module form below works directly from a source
checkout as well. The equivalent installed command is `ripii-world`.

```bash
# Five architectures, three initialization seeds, all held-out regimes.
python -m ripii.world benchmark --output runs/world_experiment --steps 300 --seeds 3 7 11
python -m ripii.world verify runs/world_experiment
python -m ripii.world capture runs/world_experiment --output research/results/development/world_experiment_capsule.json
python -m ripii.world verify-capsule research/results/development/world_experiment_capsule.json

# Open an actual interactive desktop window using the selected trained checkpoint.
python -m ripii.world demo --checkpoint runs/world_experiment/multiscale_continuous/seed_3/best.pt

# Render the same demo without a display.
python -m ripii.world demo --checkpoint runs/world_experiment/multiscale_continuous/seed_3/best.pt --export runs/world_preview.png
```

The interactive demo uses Matplotlib's native GUI backend. Left-drag an object in the
simulator panel to move its initial position; right-drag to set its velocity. Select
an object and use force sliders to apply a sustained intervention. Play/pause,
rewind, new-scene, object-count, and palette controls all work on the loaded weights.
White outlines mark the selected object. The second panel is a learned rollout, not
an invocation of the physics simulator. More than four objects is outside the
training object-count range. Palette changes affect rendering only; this is a
state-input model, so visual invariance is not a learned achievement.

## What runs

- **Simulator:** 2D damped soft-disc collisions, wall contacts, drag, and per-object
  external forces, integrated through four substeps per 0.05-second observation.
- **States:** x, y, vx, vy, radius, mass. Variable object counts use explicit masks.
  Masked slots exert no force and do not enter learned aggregation or VQ losses.
- **MLP:** padded flat scene/action features and a learned global transition.
- **Graph:** two local learned object-interaction layers with relative position,
  relative velocity, and distance features.
- **Transformer:** two attention layers with object masks and no positional index
  embeddings. Random object permutations are included in training for all models.
- **Global pool:** local object interactions plus one parameter-matched global mean
  scene context and object-level refinement. It is the direct control for whether
  multiscale grouping adds value beyond ordinary global context.
- **Multiscale:** local interactions, learned soft grouping into coarse nodes,
  coarse interaction, unpooling, and object-level refinement. Both interaction paths
  are trained through the rollout objective. This is a new world-model architecture,
  not a claim that the legacy projective stack has been validated.
- **Shared prior:** all models add learned corrections to a constant-velocity
  integration step and preserve radius/mass. No learned model calls the simulator.
- **Loss:** fixed weights: one-step state error plus half the mean multi-step state
  error. Position coordinates receive weight 4, velocity coordinates weight 1.
  Optional quantization has its separately declared fixed auxiliary weight.

## Selection and generalization

The protocol JSON is written before training. All variants see identical train and
validation trajectories. Seeds vary initialization and minibatch sampling on that
fixed dataset. Nearest hidden widths approximate the multiscale model's parameter
count; exact counts and relative errors are recorded. Equal optimizer updates do
not imply equal FLOPs or wall-clock cost, so training and inference time are reported.

Checkpoints are selected on validation position RMSE plus 0.25 times velocity RMSE.
Test trajectories are only generated and evaluated after training all candidates:

| Split | Difference from training |
|---|---|
| Test | Independent scenes with 2–4 objects |
| More objects | 5–8 objects by default |
| Composition | Large/heavy objects, absent from training; size and mass individually occur in training |
| Fast | Higher initial velocity distribution |

Training uses 16-step trajectories and a 4-step rollout objective by default.
Evaluation uses 32-step rollouts and reports endpoint errors at 1, 4, 16, and 32
steps, aggregate position/velocity RMSE, arena escapes, worst-scene IDs, and runtime.
Three explicit analytic references are included: state persistence, constant
velocity, and force/mass kinematics without collisions, walls, or drag. This prevents
the learned models from being compared only with one ambiguously weak baseline.
Reports retain negative outcomes.

Multiscale evaluations additionally report assignment entropy, normalized entropy,
effective group count, minimum/maximum group occupancy, and temporal assignment
change. FSQ evaluations report level utilization and effective levels; VQ evaluations
report coarse/fine usage, entropy, perplexity, and residual energy. These diagnostics
detect collapsed or decorative bottlenecks but do not by themselves establish better
forecasting.

The default advancement rule requires multiscale to improve average OOD position
RMSE over the graph model by at least 5%, with no more than 5% in-distribution
regression, on every seed and at least three seeds. This is a development gate, not
statistical significance. Do not change the rule after inspecting a test result.
A new hypothesis needs a fresh output directory, protocol, and preferably fresh data
seed. Source hashes and artifact hashes are retained for auditability. Benchmark
runs snapshot package source and available packaging/lock metadata before training;
the initial `world_v1` runs predate source-snapshot retention and the global-pool
control.

The fresh source-snapshotted `runs/world_v2_controls` execution used seeds 13, 17,
and 19 and all five continuous-latent architectures. Its 124-file manifest verifies,
but the predeclared decision is `no_advance`: graph has the lowest mean position RMSE
on the test split and on each OOD split, and multiscale passes the paired advancement
rule on only one of three seeds. Multiscale's mean OOD error is lower than the direct
global-pool control, but that direction is not consistent on every seed. This is a
negative development result, not a reason to alter the gate or claim hierarchy value.

The subsequent `runs/world_v3_convergence` protocol changed the data seed, used five
new seeds, increased the ceiling to 1,000 updates, and focused on graph, global-pool,
and multiscale. The 124-file manifest verifies and the decision remains
`no_advance`. Graph mean position RMSE is 0.0904 in distribution and
0.1164/0.1203/0.1755 on more-objects/composition/fast; multiscale is
0.0983 and 0.1216/0.1301/0.2130. Multiscale loses the graph OOD comparison on
every seed. Its mean OOD advantage over global-pool is only 1.58%, paired direction
is mixed, and mean ID error is worse. This is the current strongest local evidence
and it does not support advancing the hierarchy.

Because `runs/` is intentionally ignored, the compact self-checksummed capsule at
`research/results/development/world_v3_convergence_capsule_v2.json` retains the
protocol, full summary, report, and complete-run manifest without pretending that
the omitted checkpoints can be re-evaluated from the capsule alone.

## Long-range coupling study

The optional `--global-coupling` coefficient adds symmetric harmonic attraction
between all live object pairs. Pair momentum is conserved without actions, drag, or
walls; zero exactly recovers the original simulator. This creates a controlled need
for nonlocal information. The decisive simple control is global mean pooling, not the
radius-limited graph alone.

The prospectively written v4 development intervention compared coupling 0 versus 1
using graph, global-pool, and multiscale models; seeds 53, 59, 61; and 100 updates. It
returned `no_advance`. On coupled larger-object scenes, mean RMSE was 0.2043 graph,
0.1867 global-pool, and 0.1846 multiscale. Multiscale's 1.09% mean advantage over
global-pool was mixed by seed and far below the all-seed 5% rule. The coupling-specific
interaction was -0.09 percentage points (exact sign-flip p=1.0).

A separate 300-update coupled run with seeds 103, 107, 109 also returned `no_advance`.
Global-pool had the best mean IID and all three mean OOD errors; multiscale versus
global-pool differences were not directionally consistent. Both studies are synthetic
development evidence, not a confirmatory novelty claim.

```bash
python scripts/run_coupling_study.py --output runs/world_v4_coupling
python scripts/run_coupling_study.py --verify runs/world_v4_coupling
python scripts/run_coupling_study.py --capture runs/world_v4_coupling \
  --capsule-output research/results/development/world_v4_coupling_capsule.json
python scripts/run_coupling_study.py --verify-capsule \
  research/results/development/world_v4_coupling_capsule.json
python scripts/analyze_failures.py runs/world_v3_convergence \
  --output runs/world_v3_failure_analysis.json
```

## Train and resume one model

```bash
python -m ripii.world train --output runs/world_single --model multiscale --steps 300 --seed 3
python -m ripii.world train --output runs/world_single --model multiscale --steps 600 --seed 3 --resume runs/world_single/latest.pt
```

Resume restores optimizer, minibatch RNG, Torch RNG, selected validation score, and
completed-update count. Keep all experiment settings identical except the total
step budget. Continuing a completed budget fails clearly instead of repeating an
update. `best.pt` is validation-selected; `final.pt` is the terminal state.

## Compare quantization only when useful

Continuous latents are the default. Both optional bottlenecks execute and train:

```bash
python -m ripii.world benchmark --output runs/world_bottlenecks --models multiscale --bottlenecks continuous fsq vq --steps 300 --seeds 3 7 11
python scripts/qualify_quantizer.py --seed 17 --steps 300
```

FSQ uses four bounded scalar coordinates and five fixed levels per coordinate with a
straight-through gradient estimator. VQ uses the existing hierarchical residual
quantizer with 16 coarse and 16 fine entries. Neither option is assumed helpful.
The world-model benchmark evaluates downstream rollout error; the existing isolated
quantizer qualification tests a different, known-cluster task. Neither experiment
is a compression bitrate or language-tokenization benchmark.

## Boundaries and next experiments

There is no adaptive compute claim: the multiscale path executes all its layers.
Node-order equivariance is tested for graph, Transformer, and multiscale models;
rotation/translation equivariance is not imposed. Simulator contacts are soft, not
an exact rigid-body solver. Long rollouts can fail visibly. External visual data,
learned perception, physical hardware, and independent reproduction remain future
experiments rather than implemented capabilities.

The package provides executable software and reproducible local tests. An empirical
advantage cannot be guaranteed by implementation; use the generated report to decide
whether a mechanism deserves more training.
