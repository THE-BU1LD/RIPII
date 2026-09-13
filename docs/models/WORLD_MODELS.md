# Model Card: RIPII World Models

## Scope

RIPII world models predict the next exact object state from the current state, applied
force, and live-object mask. They exclude perception and uncertainty estimation. They
are research prototypes, not safety- or control-certified models.

## Implemented families

- `mlp`: fixed-capacity flattened scene baseline.
- `graph`: local interaction network.
- `transformer`: permutation-compatible self-attention baseline.
- `global_pool`: local interaction plus global mean context.
- `multiscale`: learned soft groups, coarse all-pairs interaction, and refinement.
- `equivariant`: continuous E(2)-equivariant baseline using scalar invariants and
  vector combinations; it does not support FSQ/VQ bottlenecks.

The learned families preserve object-slot permutation equivariance. Only `equivariant`
is architecturally translation-, rotation-, and reflection-equivariant; every evaluated
model now reports numerical translation and quarter-turn diagnostic errors.

## Training and limitations

Training uses autoregressive windows, validation-selected checkpoints, gradient
clipping, and a coordinate-weighted state loss. Existing evidence uses a four-step
training rollout and up to 32-step evaluation. The generic learned predictor contains
historical output scales and is not guaranteed to preserve energy, momentum, contacts,
or walls. Radius and mass are copied unchanged.

Reports must include parameter count, training budget, validation rule, data/content
hashes, position/velocity RMSE, physical diagnostics, symmetry diagnostics, and failure
strata. Parameter matching does not imply FLOP, memory, wall-clock, convergence, or
tuning-budget matching.

## Evidence status

Current retained studies are synthetic development evidence and support a bounded
negative conclusion: the multiscale model has not beaten the strongest included flat
controls. The new equivariant baseline has code/tests but no publication-grade result
yet. Do not cite implementation availability as empirical superiority.

