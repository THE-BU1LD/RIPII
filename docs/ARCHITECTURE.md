# Architecture and data flow

The legacy path is `synthetic.py -> encoder -> projective stack -> latent graph/action
-> optional hierarchical VQ -> decoder -> legacy losses`. It is retained for frozen
pilots and should not be extended without a new objective hypothesis.

The world path is `physics.py -> Experiment -> WorldModel -> validation selection ->
held-out rollout evaluation -> JSON report/manifest/capsule`. `WorldModel` supplies MLP,
graph, Transformer, global-pool, multiscale, and continuous E(2)-equivariant processors
under one state/action/mask contract. The equivariant baseline constructs vector updates
only from relative vectors, relative velocities, applied force, and learned scalar
invariants. Local interaction layers gather only declared neighbor messages instead of
materializing dense hidden pair features. `failure_analysis.py` stratifies true
pre-transition regimes without using
predictions to choose categories. `statistics.py` treats paired training seeds as units.

`data.py` owns the adapter registry, tensor contract, units, license metadata, split
policy, and deterministic content fingerprint. `protocol.py` turns an experiment
mapping into immutable canonical JSON only after schema validation. New world benchmarks
write `status.json` through `planned -> running -> complete` or `failed`; the completed
state is bound into the run manifest. Self-checksummed analysis artifacts remain separate so
post-result work does not mutate the original run. `profiling.py` isolates machine-local
timing and lower-bound compute diagnostics from scientific evaluation metrics.
Evaluation also separates coordinate error from kinetic-energy, contact penetration,
wall penetration, passive-step momentum, and numerical translation/rotation diagnostics.

Legacy and world trainers intentionally remain separate because they optimize different
objects and claims. Shared utilities are centralized only where contracts genuinely match.
