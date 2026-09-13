# Learned hierarchy does not improve a controlled synthetic object-dynamics benchmark

## Abstract

We test whether learned soft grouping improves object-state dynamics prediction over
simpler controls. In a fixed synthetic soft-disc simulator, a parameter-matched
multiscale graph model is compared with flat graph and global-context models using
validation-selected checkpoints and paired seeds. A five-seed, 1,000-update development
study fails its predefined advancement rule: multiscale loses the flat graph OOD
comparison on every seed. A subsequent two-domain NRI external-simulator development
study also failed its hierarchy advancement gate. This is a bounded negative result,
not a novelty claim. The present evidence favors simpler controls.

## Scope, method, and protocol

The input is exact object state and applied force; perception is excluded. Mathematical
contracts are in `research/MATHEMATICAL_SPEC.md`. Training uses a four-step autoregressive
objective and validation-only checkpoint selection. Test regimes change scene identity,
object count, held-out radius/mass composition, and initial velocity. Persistence,
constant-velocity, force-kinematic, MLP, Transformer, global-pool and flat graph controls
separate hierarchy from capacity and generic global context. The v3 study uses a new
data seed and five new training seeds. It is development evidence because it was not
externally preregistered and uses one simulator; the later NRI extension is also
development rather than confirmatory evidence.

## Results

{{WORLD_V3_RESULTS}}

These rounded values are descriptive. The capsule, rather than this rendered table, is
the numerical source of record.

## Failure analysis and limitations

Unconstrained soft groups may mix bodies that do not share useful physical locality;
the local graph already represents the dominant interaction prior. The benchmark uses
known states, one dynamics law, limited object counts, and small seed samples. Parameters
and update counts are approximately matched, but FLOPs are not. No real-world data,
learned perception, adequately powered cross-dataset analysis, or independent
reproduction is available. Thus the study falsifies only the current implementation at
the tested scale.

Post-result failure localization over the five retained seeds shows the deficit is not
specific to collisions: multiscale is worse than graph in contact, near-contact, forced,
and free-flight subsets, with the largest relative deficits under faster motion.

## External-simulator development extension

We subsequently evaluated the same five neural control families on commit-pinned NRI
Springs and Charged simulators, using disjoint generated train/validation/test
trajectories and three paired seeds per domain. Multiscale met the predeclared 5%
improvement over global pooling in only one of six domain-seed cells and was not the
best model on either domain. Transformer was best on Springs; MLP was the best neural
model on Charged, where constant velocity was better than every neural model. This
extends the negative evidence beyond the bundled simulator, but locally generated NRI
trajectories, three seeds, unmatched compute, and a development protocol cannot support
population or real-world claims.

## Targeted long-range intervention

We introduced a symmetric all-pairs harmonic force as a prospectively specified
development intervention. It preserves momentum and makes nonlocal information
relevant. The primary control is global pooling rather than the deliberately local
graph. Across three seeds, multiscale improved coupled more-objects RMSE over global
pool by only 1.09% on average with mixed directions; the coupled-minus-local relative
advantage was -0.09 percentage points (exact sign-flip p=1.0). A separate 300-update,
new-seed follow-up placed global pooling first in mean IID and all OOD regimes. Thus
the observed boundary condition is access to global information, not learned hierarchy.

## Objective simplification and efficiency boundary

In a separately frozen 39-run local study, reconstruction+KL outperformed every
one-term addition and the complete legacy objective on all three paired seeds at 30
updates. The complete objective increased reconstruction error by 26.4%, 60.0%, and
89.5%. A post-training diagnostic found the invariance term's mean weighted gradient
norm was 6.04 versus 1.32 for reconstruction and found a negative cosine on at least
one seed for 63/78 term pairs. These diagnostics motivate simplification but do not
prove which conflict caused the result.

Machine-local profiling records parameters, repeated latency, throughput, checkpoint
size, and profiler-recognized operations. Recognized operation counts differ despite
similar parameter counts, while repeated sessions change latency ordering. We therefore
make neither an efficiency claim nor a compute-matched performance claim.

## Related work and conclusion

Object-centric interaction networks [1], graph-network simulators [2], neural relational
inference [3], E(n)-equivariant graph networks [4], multiscale mesh simulators [5,6],
constraint-based graph simulators [7], VQ, and FSQ [8] all predate this repository; see
`research/NOVELTY_AUDIT.md`. The current result does not support advancing RIPII's
hierarchy. A submission-grade negative result would require multiple public datasets,
adequate paired seeds, compute matching, and analysis that localizes when grouping
harms prediction.

After the frozen studies, the repository added an E(2)-equivariant continuous baseline
and explicit symmetry, energy, contact, wall, and conditional-momentum diagnostics.
Those additions have tests but no result in this manuscript and do not alter the frozen
negative conclusion.

## Reproducibility statement

The code records source hashes, complete experiment configuration, seeds, validation
selection, per-seed raw metrics, timing, checkpoints, success state, and artifact hashes.
Canonical commands are in the README. The self-checksummed compact capsule omits
checkpoints; full local runs are ignored mutable storage. A SHA-256 digest detects
corruption but does not authenticate an author. No license has been selected, so reuse
rights are not granted by this manuscript.

## Ethics, authorship, and disclosures

The bundled studies use synthetic trajectories and contain no human-subject or personal
data. Authorship, affiliations, author contributions, funding, conflicts of interest,
and an owner-approved software/data license remain unresolved release blockers. They
must be supplied by authorized humans and are not inferred by repository automation.

## References

1. Battaglia et al. Interaction Networks for Learning about Objects, Relations and
   Physics. NeurIPS, 2016.
2. Sanchez-Gonzalez et al. Learning to Simulate Complex Physics with Graph Networks.
   ICML, 2020.
3. Kipf et al. Neural Relational Inference for Interacting Systems. ICML, 2018.
4. Satorras et al. E(n) Equivariant Graph Neural Networks. ICML, 2021.
5. Fortunato et al. MultiScale MeshGraphNets. arXiv:2210.00612, 2022.
6. Cao et al. Accelerating Mesh-based Simulations with Bi-Stride Multi-Scale Graph
   Neural Networks. ICML, 2023.
7. Rubanova et al. Constraint-based Graph Network Simulator. ICML, 2022.
8. Mentzer et al. Finite Scalar Quantization: VQ-VAE Made Simple. arXiv:2309.15505,
   2023.
