# Abstract {-}

Local graph neural simulators must spend depth to propagate information across a large
physical domain, whereas dense global processors can be expensive and may ignore
physical structure. Existing hierarchical simulators address this problem, but a learned
coarse path can also become an unconstrained global shortcut. We introduce RIPII-MR, an
adaptive multiresolution particle simulator whose geometric restriction is a mass-aware
partition of unity, whose internal pair updates are Euclidean-equivariant and conserve
linear and angular momentum, and whose coarse contribution is weighted by a causal hard
gate. The current implementation still computes that coarse path on every forward call.
The method predicts physical impulses and uses an explicit semi-implicit update;
it does not use the vector quantization, latent projection stack, direct position bypass,
or multi-loss objective of the failed legacy RIPII system. Construction-level tests
verify permutation and E(2) equivariance, restriction mass preservation, internal
momentum and torque preservation, action semantics, gradient reachability, and complete
checkpoint/benchmark execution. The prospectively specified 75-run synthetic development
matrix completed with a `no_advance` decision: none of 60 paired candidate/control
comparisons passed the frozen gate, and mean OOD relative improvement was negative
against every control. Accordingly, this manuscript makes no positive accuracy, novelty,
significance, external-validity, efficiency, or state-of-the-art claim. Its current
contribution is a falsifiable method and a preserved negative development result.

# Introduction

Particle and mesh simulators represented by graphs combine physical locality with
learned interaction rules. Graph Network-based Simulators (GNS) demonstrated that
single-step message-passing models, trained with state perturbations, can produce long
rollouts and transfer to larger particle systems [1]. MeshGraphNets extended the same
design pattern to unstructured simulation meshes [2]. These methods face a basic
communication problem: on a high-resolution graph, physically distant nodes may require
many message-passing steps to influence one another.

Multiscale graph simulators respond by passing messages on coarsened graphs. MultiScale
MeshGraphNets and Bi-Stride GNNs already establish that fine/coarse processing can improve
accuracy and efficiency in compatible mesh settings [3,4]. Hierarchy alone is therefore
not a novel contribution. The open question addressed here is narrower: can a hierarchy
be made adaptively contribution-gated, symmetry-respecting, conservative, and
mechanistically accountable, so that its coarse correction contributes only when the
local receptive field is insufficient?

The question is motivated by a failure. The legacy RIPII hierarchy lost to flat graph and
global-context controls, and its full reconstruction objective lost on all five paired
seeds to a simplified objective and to a plain autoencoder. Its soft assignments were
active but not useful. This rules out a cosmetic retuning story and motivates a clean
method boundary.

The prespecified method components and study were:

1. a mass-aware geometric restriction and prolongation operator with partition-of-unity
   and exact mass-accounting tests;
2. an E(2)-equivariant impulse model whose internal fine and coarse corrections preserve
   total linear and angular momentum by construction;
3. a causal hard contribution gate trained with a straight-through estimator; the
   current implementation does not skip coarse computation;
4. a frozen multi-data-seed experiment and ablation program that requires improvement
   over equivariant, local, global-pool and legacy multiscale controls.

Items 1--3 are implemented and construction-tested. Item 4 was executed and failed its
predeclared gate. The structural properties therefore remain engineering results, while
the predictive development hypothesis is rejected for this architecture version.

# Related Work and Novelty Boundary

GNS provides the principal local particle-message-passing baseline [1]. EGNN provides a
computationally economical construction for translation, rotation, reflection and
permutation equivariance [5]. Work on E(3)-equivariant Lagrangian fluid models shows that
symmetry-aware interactions can improve physical accuracy, although computational cost
can increase [6]. Constraint-based GNS replaces direct next-state prediction with a
learned consistency objective and iterative solve, enabling test-time computation and
new constraints [7]. Neural SPH injects solver-derived pressure, viscosity and forcing
structure to address particle clustering and long-rollout instability [8].

RIPII-MR does not claim priority for graph simulation, equivariance, conservation,
coarsening, adaptive computation or multiscale message passing individually. A future,
materially revised method could test the jointly constrained mechanism: true conditional
routing of an equivariant, conservative geometric hierarchy, supported by a superior
measured accuracy-compute frontier and targeted mechanism interventions. The current
implementation and negative matrix do not establish that claim.

# Problem Formulation

At time step \(t\), particle \(i\) has position \(x_i^t\in\mathbb{R}^2\), velocity
\(v_i^t\in\mathbb{R}^2\), positive mass \(m_i\), radius \(r_i\), and external force
\(f_i^t\). A Boolean mask distinguishes live particles from padding. The model predicts
the next state while copying time-invariant particle properties exactly.

The preregistered development hypothesis was:

> When the physical correlation length exceeds the receptive field of a local graph,
> RIPII-MR improves long-rollout error over matched local, global-context, equivariant and
> legacy multiscale controls, while preserving physical diagnostics and a competitive
> accuracy-compute frontier.

The matching local hypothesis is a negative control: when interactions fit inside the
local receptive field, hierarchy need not help.

# Method

## Invariant encoding and local interaction

The node encoder consumes scalar invariants: squared speed, squared action magnitude,
the velocity-action inner product, radius and mass. For an undirected local edge
\((i,j)\), a symmetric scalar network receives commutative combinations of node features
and the invariants

\[
\|x_i-x_j\|^2,\quad \|v_i-v_j\|^2,\quad
(x_i-x_j)^\top(v_i-v_j).
\]

It predicts a bounded scalar \(\alpha_{ij}=\alpha_{ji}\). The pair impulse is central,

\[
J_{ij}=\alpha_{ij}\frac{x_i-x_j}{\|x_i-x_j\|},\qquad J_{ji}=-J_{ij}.
\]

This produces equal-and-opposite internal impulses without a conservation penalty.

## Geometric restriction

Each scene selects at most \(K\) anchors by farthest-point traversal from the geometric
center. Each particle connects to its nearest anchor memberships. Distance-kernel weights
are nonnegative and normalized over those memberships:

\[
w_{ik}\ge 0,\qquad \sum_k w_{ik}=1.
\]

The restricted mass and state are

\[
M_k=\sum_i w_{ik}m_i,\qquad
\bar x_k=M_k^{-1}\sum_iw_{ik}m_ix_i,\qquad
\bar v_k=M_k^{-1}\sum_iw_{ik}m_iv_i.
\]

The same normalized mass shares restrict scalar features. Empty groups remain masked.
The coarse graph is a symmetric sparse k-nearest-neighbor graph rather than a dense
all-to-all feature tensor.

## Conservative prolongation

Coarse nodes exchange the same form of central antisymmetric impulse. If \(\Delta P_k\)
is the total coarse impulse, particle \(i\) receives

\[
\Delta p_i^c=\sum_k \frac{w_{ik}m_i}{M_k}\Delta P_k.
\]

Because the shares sum to one within each coarse group, prolongation preserves the total
coarse impulse. Distribution over extended groups can introduce fine-level torque, so a
closed-form projection removes only the rotational component while preserving net linear
momentum. Let \(r_i=x_i-x_{cm}\), \(I=\sum_i m_i\|r_i\|^2\), and
\(\tau=\sum_i r_i\times\Delta p_i^c\). The corrected impulse is

\[
\widetilde{\Delta p_i^c}=\Delta p_i^c-\frac{\tau}{I}m_i r_i^\perp.
\]

Degenerate zero-inertia scenes use a finite denominator guard and have zero observable
torque about their common position.

## Adaptive routing and integration

A scalar router uses the mass-weighted scene representation and mean local-impulse scale
to predict \(q\in(0,1)\). The coarse route is active when \(q\ge q_0\). Training uses a
straight-through hard gate; evaluation uses the deterministic threshold. The model logs
probability, activation rate and fine/coarse edge counts. Restriction and coarse message
passing occur before this decision, so the gate controls the coarse contribution rather
than whether coarse computation is performed.

Fine and routed coarse impulses are projected once more to remove accumulated numerical
torque. External action is applied separately as \(\Delta p_i^{ext}=\Delta t f_i\). The
semi-implicit update is

\[
v_i^{t+1}=v_i^t+(\Delta p_i^{int}+\Delta t f_i^t)/m_i,\qquad
x_i^{t+1}=x_i^t+\Delta t v_i^{t+1}.
\]

Unlike legacy RIPII, there is no arbitrary direct position correction.

# Structural Guarantees

The implemented operations are permutation equivariant for generic non-tied anchor
selection. Distances and scalar inner products are invariant to E(2); central impulses
and action forces transform as vectors. Restriction weights are geometric scalars, so
restriction and prolongation commute with translations, rotations and reflections. Pair
antisymmetry makes total internal impulse zero. Central fine/coarse interactions are
torque-free before prolongation, and the explicit projection makes prolonged internal
torque zero.

The implementation test suite checks these statements numerically under random masks,
positions, velocities and actions. A limitation remains: exact symmetric anchor ties need
an explicitly specified set-valued or canonical tie treatment before claiming universal
permutation equivariance as a theorem.

# Training Objective

The candidate uses the existing direct dynamics training system: a weighted state error
on position and velocity, one-step plus mean scheduled multi-step rollout loss, optional
training-state noise, validation-only checkpoint selection, gradient clipping and exact
resume state. RIPII-MR has no VQ loss or learned uncertainty-weighted loss bank.

The development run schedules the rollout horizon from one to eight steps over the first
600 updates and uses state noise 0.002. These are frozen development choices, not
universally optimal settings. Their effects must be isolated by later training-schedule
ablations.

# Experimental Protocol

The first matrix contains three independent synthetic data seeds, five paired
model/minibatch seeds and five learned models, totaling 75 training runs. All models see
the same train/validation/test scenes within a data seed. Hidden widths are selected to
fall within 5% of a shared parameter-count target. The primary endpoint averages position
RMSE over more-object, held-out-composition and faster-motion splits. IID error, velocity,
energy, contact/wall penetration, passive momentum, symmetry, stability, latency and
mechanism diagnostics are secondary.

RIPII-MR advances only if it improves the OOD endpoint by at least 5% against every
included control on every paired cell while regressing IID position RMSE by no more than
5%. This strict rule is appropriate for development triage but is not population
inference. A later confirmatory design must use external public data, independent
trajectory/scene units, cluster-aware uncertainty, compute-matched tuning and an untouched
test evaluation.

# Required Ablations

The full study must isolate: local versus coarse communication; fixed versus adaptive
routing; geometric versus random grouping; conservative versus unprojected
prolongation; coarse-neighbor count; group count; local radius/depth; training noise;
rollout curriculum; integrator choice; parameter matching; and equal-wall-time matching.
Counterfactual evaluation should disable the coarse path only on routed scenes and shuffle
coarse messages between matched scenes. Spectral error by spatial wavelength and error
by interaction distance are required mechanism outcomes.

# Dataset Program

The internal soft-disc generator is a falsification environment, not external evidence.
The next valid target is LagrangeBench, which supplies public Lagrangian fluid datasets,
official GNS/EGNN/SEGNN baselines and rollout metrics [9]. The existing 2,708-particle
LDC data cannot be truncated into an eight-object tensor. Publication evidence requires a
variable-size sparse adapter that preserves particle histories, boundary types, units and
official splits. Compatible mesh tasks should then compare faithfully against
MeshGraphNets, MultiScale MeshGraphNets and Bi-Stride implementations.

# Results

The new model first passed a two-update, one-seed, five-model end-to-end smoke run whose
79 artifacts verified. That run tested plumbing only and is excluded from the scientific
result below.

The frozen development matrix then completed all 75 learned-model runs across three data
seeds, five model/minibatch seeds and five models. Its retained analysis returned
`no_advance`:

| Control | Paired cells | Mean OOD relative improvement | All cells pass |
|---|---:|---:|---:|
| equivariant | 15 | -0.1596 | false |
| graph | 15 | -0.7824 | false |
| global pool | 15 | -1.1909 | false |
| legacy multiscale | 15 | -1.2134 | false |

Thus none of the 60 candidate/control comparisons passed the frozen requirement of at
least 5% OOD improvement without more than 5% IID regression. The negative signs mean
that RIPII-MR was worse on the mean OOD endpoint for every included control. These are
synthetic development results. The three data-seed means, not the nested model-seed
cells, are the appropriate independent-unit view, and the study is not powered
confirmation. No positive effect, external validity or population-level superiority
claim follows.

The legacy negative findings remain relevant motivation: unconstrained learned grouping
did not beat local/global controls, and the full legacy objective underperformed simpler
objectives. The RIPII-MR result does not reverse those findings.

# Limitations and Failure Criteria

The current implementation uses pairwise distance construction and therefore does not yet
establish subquadratic end-to-end neighbor search. Farthest-point anchor selection has a
tie ambiguity in exactly symmetric configurations. The synthetic generator is small,
two-dimensional and lacks empirical measurement noise. Parameter matching does not imply
equal FLOPs or wall time. The routing threshold and group count are frozen development
choices. Moreover, the current forward pass computes restriction, coarse edges and
coarse impulses before the hard route gates their contribution. It therefore implements
contribution gating rather than compute-skipping conditional execution. No external
result, independent reproduction, positive effect or efficiency benefit exists.

Failure of the 75-run development gate was the predeclared stop condition, and the gate
failed. This architecture version is therefore terminated rather than advanced to
external benchmarking or confirmation. Any successor must be identified as a new
version with a newly frozen protocol; it must not overwrite this result.

# Reproducibility

The runner wrote a machine-readable protocol before training, used deterministic data
and model seed domains, selected checkpoints on validation only, snapshotted source in
each child benchmark, preserved run state and exceptions, verified child manifests, and
emitted a content-hashed outer manifest. The retained matrix should be checked without
overwriting it:

```bash
.venv/bin/python scripts/run_ripii_mr_matrix.py \
  --output runs/ripii_mr_development_v1 --verify-only
```

The source-of-truth result is `runs/ripii_mr_development_v1/report.md`, backed by
`summary.json`, `status.json`, `protocol.json` and `manifest.json` in the same directory.

# References {-}

1. Sanchez-Gonzalez et al. *Learning to Simulate Complex Physics with Graph Networks.* ICML, 2020.
2. Pfaff et al. *Learning Mesh-Based Simulation with Graph Networks.* ICLR, 2021.
3. Fortunato et al. *MultiScale MeshGraphNets.* 2022.
4. Cao et al. *Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network.* ICML, 2023.
5. Satorras, Hoogeboom, and Welling. *E(n) Equivariant Graph Neural Networks.* ICML, 2021.
6. Toshev et al. *Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks.* 2023.
7. Rubanova et al. *Constraint-based Graph Network Simulator.* ICML, 2022.
8. Toshev et al. *Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics.* ICML, 2024.
9. Toshev et al. *LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite.* NeurIPS Datasets and Benchmarks, 2023.
