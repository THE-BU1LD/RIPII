# Learned hierarchy does not improve a controlled synthetic object-dynamics benchmark

## Abstract

We test whether learned soft grouping improves object-state dynamics prediction over
simpler controls. In a fixed synthetic soft-disc simulator, a parameter-matched
multiscale graph model is compared with flat graph and global-context models using
validation-selected checkpoints and paired seeds. A five-seed, 1,000-update development
study fails its predefined advancement rule: multiscale loses the flat graph OOD
comparison on every seed. This is a bounded negative result, not external validation or
a novelty claim. The present evidence favors the simpler graph model.

## Scope, method, and protocol

The input is exact object state and applied force; perception is excluded. Mathematical
contracts are in `research/MATHEMATICAL_SPEC.md`. Training uses a four-step autoregressive
objective and validation-only checkpoint selection. Test regimes change scene identity,
object count, held-out radius/mass composition, and initial velocity. Persistence,
constant-velocity, force-kinematic, MLP, Transformer, global-pool and flat graph controls
separate hierarchy from capacity and generic global context. The v3 study uses a new
data seed and five new training seeds. It is development evidence because it was not
externally preregistered and uses one simulator.

## Results

Mean ID position RMSE is 0.0904 for graph, 0.0920 for global pool, and 0.0983 for
multiscale. Graph also has the lowest mean error in every OOD regime. Multiscale's mean
OOD relative improvement versus graph is -13.05% and it loses all five paired seeds.
Versus global pool, its mean OOD advantage is 1.58%, but paired direction is mixed and
ID error is 7.49% worse. Learned assignments remain active (2.77 effective groups of
four), ruling out numerical inactivity but not showing utility. Exact unrounded values
and provenance are in the self-checksummed v3 capsule cited by `EVIDENCE_LEDGER.md`.

## Failure analysis and limitations

Unconstrained soft groups may mix bodies that do not share useful physical locality;
the local graph already represents the dominant interaction prior. The benchmark uses
known states, one dynamics law, limited object counts, and small seed samples. Parameters
and update counts are approximately matched, but FLOPs are not. No real-world data,
learned perception, external simulator, scaling study, or independent reproduction is
available. Thus the study falsifies only the current implementation at the tested scale.

Post-result failure localization over the five retained seeds shows the deficit is not
specific to collisions: multiscale is worse than graph in contact, near-contact, forced,
and free-flight subsets, with the largest relative deficits under faster motion.

## Targeted long-range intervention

We introduced a symmetric all-pairs harmonic force as a prospectively specified
development intervention. It preserves momentum and makes nonlocal information
relevant. The primary control is global pooling rather than the deliberately local
graph. Across three seeds, multiscale improved coupled more-objects RMSE over global
pool by only 1.09% on average with mixed directions; the coupled-minus-local relative
advantage was -0.09 percentage points (exact sign-flip p=1.0). A separate 300-update,
new-seed follow-up placed global pooling first in mean IID and all OOD regimes. Thus
the observed boundary condition is access to global information, not learned hierarchy.

## Related work and conclusion

Object-centric interaction networks, differentiable graph pooling, multiscale graph
simulators, VQ, and FSQ all predate this repository; see `research/NOVELTY_AUDIT.md`.
The current result does not support advancing RIPII's hierarchy. A submission-grade
negative result would require multiple public datasets, adequate paired seeds, compute
matching, and analysis that localizes when grouping harms prediction.

## Reproducibility statement

The code records source hashes, complete experiment configuration, seeds, validation
selection, per-seed raw metrics, timing, checkpoints, success state, and artifact hashes.
Canonical commands are in the README. The self-checksummed compact capsule omits
checkpoints; full local runs are ignored mutable storage. A SHA-256 digest detects
corruption but does not authenticate an author. No license has been selected, so reuse
rights are not granted by this manuscript.
