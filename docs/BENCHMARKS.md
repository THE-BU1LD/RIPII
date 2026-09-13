# Benchmark program

This document separates executed evidence from planned evaluation. A checked-in plan,
adapter, or test is not an experimental result.

## Executed studies

| Study | Data | Controls | Seeds | Status | Claim boundary |
| --- | --- | --- | ---: | --- | --- |
| Legacy pilots v1/v2 | RIPII synthetic representations | structured removals, plain autoencoder | 3 | no advance | synthetic development |
| World v3 convergence | soft-disc simulator | graph, global pool, multiscale | 5 | no advance | one fixed simulator |
| Global-coupling study | soft discs with/without all-pairs force | graph, global pool, multiscale | 3 | no advance | synthetic intervention |
| Objective study | RIPII synthetic representations | reconstruction+KL, each auxiliary, full objective | 3 | simpler objective wins | short-budget development |
| NRI external v1 | pinned Springs and Charged simulators | analytic + five neural models | 3/domain | no advance | external simulator, not confirmatory |

Exact protocols and capsules live under `research/protocols/` and
`research/results/development/`. Historical results predate the 0.2 mechanism fixes and
must never be presented as measurements of the corrected implementation.

## Required post-correction matrix

The next development run must establish that the corrected implementation behaves as
intended before any larger study:

| Axis | Required values |
| --- | --- |
| Projector bank | 1, 2, 4 projectors; 0, 1, 2, 3 hierarchy levels |
| Graph | disabled; top-k 1, 2, full; explicit no-self-edge control |
| Quantization | continuous; FSQ; VQ; VQ without balance loss |
| Objectives | reconstruction+KL; each corrected mechanism term; full corrected objective |
| Capacity | native width; parameter matched |
| Budget | equal updates; measured training wall time; profiler-recognized operation proxy |
| Evaluation | one-step, full rollout, IID, object-count OOD, parameter OOD, law shift |

Every cell records raw per-seed metrics, configuration, source hashes, dataset hashes,
selected checkpoint, failures, wall time, parameter count, and peak accelerator memory
where available.

## Dataset ladder

1. **Controlled diagnostics:** bundled representation generator and soft-disc simulator.
2. **Independent simulators:** pinned NRI Springs and Charged, then one separately
   maintained particle or mesh simulator with compatible licensing.
3. **Fixed public observations:** one independently collected trajectory dataset with
   immutable download hashes and documented units.

New adapters must provide provenance, version, license identifier, units, feature and
mask contracts, immutable content hashes, disjoint split policy, and deterministic
preprocessing. Dataset redistribution is allowed only after license review.

## Baseline and statistics contract

Required controls are persistence, constant velocity, force kinematics when meaningful,
MLP, flat graph, Transformer, global pool, multiscale, and one strong domain-specific
published baseline. Comparisons must report parameter-matched and compute-matched views.

Development gates may use three to five seeds only to reject weak directions. A positive
claim requires a prospectively frozen powered seed count, paired analysis within each
dataset, confidence intervals, practical-effect thresholds, multiplicity handling, and
failure-inclusive reporting. Test outcomes may be examined only after the protocol and
source snapshot are frozen.

## Post-correction overnight run

Run `./scripts/start_post_correction_overnight.sh` to calibrate and launch the frozen
synthetic development matrix: 36 conditions, five paired seeds, and 600 updates per cell. The
launcher prevents duplicate live runs and uses macOS `caffeinate`. Restarting verifies
and skips sealed mode groups; incomplete groups are preserved under timestamped names
before recomputation. Follow `runs/post_correction_v02/overnight.log` for progress.

Use `./scripts/post_correction_status.sh` for machine-readable progress and the
calibrated ETA. Completion produces paired bootstrap intervals, exact sign-flip tests,
Holm-adjusted p-values, a predeclared decision, one combined raw-row summary, and a
top-level manifest. This is development evidence, not a confirmatory or real-world result.
