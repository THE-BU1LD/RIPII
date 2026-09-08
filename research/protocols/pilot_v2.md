# RIPII corrected component pilot v2

Status: frozen local follow-up pilot protocol. It was specified after pilot v1 and is
not an independent confirmation or externally timestamped preregistration.

## Question

After correcting zero-weight adaptive-loss handling, does the complete RIPII model
show a descriptive held-out advantage over bypassing quantization or removing the
four named mechanisms on new synthetic seeds?

## Scope and exclusions

This is an engineering/research follow-up, not a publication experiment. It tests one
bundled synthetic generator. It does not establish novelty, real-world utility,
scaling, significance, or external generalization. `no_structured` removes projective
refinement, graph refinement, quantization, and latent action but retains the shared
encoder, node/fusion scaffold, and decoder; it is not a plain autoencoder and is not
parameter matched.

## Frozen design

- Config: `configs/pilot_v2.yaml`.
- Seeds: `401`, `503`, `607` (not used by pilot v1).
- Modes: `base`, `no_vq`, `no_structured`.
- Budget: 30 optimizer steps per seed and mode.
- Data: one deterministic 128-example synthetic world per seed.
- Split: deterministic 70% train, 15% validation, 15% test with disjoint indices.
- Initialization: one retained full-model initial state per seed; compatible tensors
  are copied into each mode and the training RNG is reset after construction.
- Model selection: final checkpoint only.

## Outcomes and decision

Primary outcomes are mean test reconstruction MSE and held-out ridge-probe accuracy.
The probe is fit on train features and scored once on test features.

The quantizer earns further work only if `base` has lower mean reconstruction MSE than
`no_vq`, base probe accuracy is no more than 0.05 below `no_vq`, and both base mean
effective-code fractions exceed 0.25. The full mechanism stack earns broader follow-up
only if `base` also has lower mean reconstruction MSE than `no_structured` with probe
accuracy no more than 0.05 below it. No p-value or confirmatory claim is authorized.

## Failure rules

The pilot is invalid if any run fails, a value is non-finite, split indices overlap,
the protocol/config digest differs, a declared artifact fails verification, exact
paired initialization files are absent, or a disabled quantizer reports numeric
quantizer diagnostics. Adaptive total losses are never a cross-mode ranking outcome.
