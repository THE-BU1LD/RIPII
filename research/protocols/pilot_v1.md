# RIPII component pilot v1

Status: frozen local pilot protocol. This is not an externally timestamped
preregistration and cannot establish publication-grade chronology.

## Question

On the bundled synthetic structured-data generator, does the complete RIPII model
show a descriptive held-out advantage over removing its claimed structured mechanisms?

## Scope and exclusions

This pilot tests implementation feasibility and generates hypotheses. It does not test
novelty, external generalization, real-world usefulness, compute efficiency, or
publication-level significance. The component-removal models are not parameter-matched.

## Frozen design

- Config: `configs/pilot_v1.yaml`.
- Seeds: `101`, `211`, `307`.
- Modes: `base`, `no_renorm`, `no_graph`, `no_vq`, `no_action`, `no_structured`.
- Budget: 30 optimizer steps per seed and mode.
- Data: one deterministic 128-example synthetic world per seed.
- Split: deterministic 70% train, 15% validation, 15% test with disjoint indices.
- Initialization: each seed has one full-model initial state; every compatible tensor is
  copied into every mode. Training RNG is reset after construction for paired minibatch
  order and latent sampling.
- Model selection: final checkpoint only. Validation is diagnostic; it does not choose
  a pilot outcome.

## Outcomes

Primary descriptive outcomes are test reconstruction MSE (`recon`) and held-out ridge
probe accuracy. The probe is fit on train representations and scored on test
representations. Transform consistency and codebook diagnostics are secondary.

The pilot advancement signal is present only if the base model has lower arithmetic-mean
test reconstruction MSE than `no_structured` and held-out probe accuracy no more than
0.05 below `no_structured`. Individual component removals remain descriptive. No
p-value or confirmatory claim is authorized from three seeds.

## Failure rules

The pilot is invalid if a run fails, a value is non-finite, a split overlaps, the
protocol/config digest differs, retained files are missing, or paired initialization
metadata is absent. The existing `runs/ripii_smoke` files are excluded.

## Command

The protocol-bound command is recorded in `research/results/pilot_v1/COMMAND.md` after
the protocol digest is calculated. Output paths must be fresh and retained artifacts
must share the summary directory.
