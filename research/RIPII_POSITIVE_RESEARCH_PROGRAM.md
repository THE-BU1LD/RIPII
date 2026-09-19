# RIPII Positive Research Program

## Decision

RIPII should **not** be rescued by retuning the current full autoencoder, adding more
auxiliary losses, or relabelling learned pooling as renormalization. The completed frozen
study gives the opposite evidence: the full objective (mean reconstruction MSE 0.08410)
lost on every paired seed to reconstruction+KL (0.04687) and to a plain autoencoder
(0.05573). The hierarchy also lost to simpler graph/global-context controls in the world
studies. Those negative results are assets: they sharply identify what a new contribution
must explain and beat.

The recommended research direction is a new method, provisionally **RIPII-MR**:

> **A geometry-aware, E(2)/E(3)-equivariant, conservative, adaptive multiresolution
> particle simulator.** It uses a sparse partition-of-unity hierarchy only when a learned
> residual diagnostic predicts that local message passing lacks long-range information;
> its coarse-to-fine correction is constrained to preserve physical symmetries and cannot
> act as a generic dense pooling shortcut.

This is potentially a real contribution only if it establishes all three claims below:

1. **Mechanism:** the hierarchy reduces the long-range information bottleneck, rather
   than merely adding capacity or global pooling.
2. **Physics:** equivariance and the relevant conservation laws hold by construction,
   and coarse corrections do not inject net momentum or torque.
3. **Value:** it improves accuracy--compute scaling and long-rollout stability over
   matched strong baselines on public, appropriately sized problems.

If any claim fails under the gates below, preserve the negative result and publish the
failure analysis; do not iterate architecture after seeing confirmatory test outcomes.

## Non-negotiable scientific reset

| Current premise or component | Finding | Decision |
|---|---|---|
| “Projective renormalization” | No RG map, scale transformation, fixed point, semigroup, universality result, or physical coarse variable is defined. | Remove the term *renormalization* from method claims. Call it a learned latent projection until a separate mathematical program earns more. |
| Global learned soft assignments | Active but not useful; assignments are not geometry- or symmetry-constrained. | Replace with explicit geometric restriction/prolongation satisfying verifiable algebraic conditions. |
| VQ as a central mechanism | Near-collapse in earlier pilots; no demonstrated task benefit. Straight-through optimization is biased. | Remove VQ from the core simulator. Keep it only as a separately justified compression experiment. |
| Thirteen-term adaptive objective | Terms have unequal/conflicting gradients and every tested addition hurt fixed-budget reconstruction. | Start from one direct prediction loss plus a small, motivated rollout objective. Add one term at a time only after a frozen ablation. |
| Fixed residual constants (0.5, 0.05) | Hide units, condition scale and give the model an arbitrary position bypass. | Predict acceleration/impulse in normalized physical units and integrate with a specified numerical scheme. |
| Dense/local hierarchy | Current neighbor search remains quadratic and the group processor is all-to-all. | Make asymptotic and measured compute improvement a primary claim, not an implementation detail. |
| Internal small-disc simulator | Useful for falsification, insufficient for novelty or external validity. | Retain as a diagnostic suite; move the main result to public particle/mesh physics benchmarks. |

## Proposed hypothesis

### H1: conditional multiresolution advantage

For dynamics in which the physical correlation length is substantially larger than the
local message radius, RIPII-MR achieves lower paired per-trajectory long-rollout error
than the strongest validation-selected non-hierarchical baseline at equal measured
training and inference budgets. It preserves the selected physical diagnostics and gains
grow with problem resolution or interaction range.

This is deliberately conditional. It does **not** claim that hierarchy improves every
small, local simulator. A flat graph should remain competitive or win when the receptive
field already covers the physics; that is an expected negative control.

### What would make this novel

Novelty cannot be “hierarchical GNN for simulation”: MultiScale MeshGraphNets already
uses fine/coarse message passing to overcome long graph distances, and Bi-Stride GNNs,
multigrid-inspired simulators, and newer hierarchical mesh methods occupy that space.
The contribution must instead be a sharply specified combination that prior work does
not provide and that is isolated experimentally:

- **adaptive routing certificate:** compute the coarse path only for scenes/layers where
  a causal long-range residual score exceeds a fixed threshold, with a calibrated
  accuracy-versus-FLOP curve;
- **conservative coarse correction:** every fine-level correction is an antisymmetric
  pair impulse or is projected into the zero-net-force/zero-net-torque subspace;
- **equivariant restriction/prolongation:** assignments depend only on relative
  geometry and invariant/scalar features, while vector features are transported in an
  equivariant local frame;
- **scale-consistency objective:** coarse prediction agrees with a known physical
  restriction of the fine prediction, rather than requiring arbitrary latent matching;
- **theorem plus test:** prove the exact symmetries/conservation that the implementation
  preserves and test them to floating-point tolerance.

The paper title and claims should center the *conditional accuracy--compute and
long-range generalization result*, not a broad claim about representation learning.

## Target mathematical formulation

Let particles have position \(x_i\in\mathbb R^d\), velocity \(v_i\), mass \(m_i>0\),
type/boundary features \(c_i\), and local graph \(\mathcal E_r\). Let anchor points
\(a_k\) be selected geometrically (farthest-point sampling, learned only after a fixed
geometric candidate set, or mesh cells). Define nonnegative compact-support weights
\(w_{ik}=\phi(\|x_i-a_k\|/h_k)\), normalized so \(\sum_k w_{ik}=1\).

Restriction must use physical weights:

\[
M_k=\sum_i w_{ik}m_i,\qquad
\bar x_k=M_k^{-1}\sum_i w_{ik}m_i x_i,\qquad
\bar v_k=M_k^{-1}\sum_i w_{ik}m_i v_i.
\]

The coarse graph is sparse--radius, kNN, or multipole edges--not all-to-all. A local
equivariant processor predicts pairwise fine and coarse impulses. For each active pair,
require \(J_{ij}=-J_{ji}\); update momenta by \(p_i^{t+1}=p_i^t+\sum_jJ_{ij}\). A
coarse correction should be distributed through \(w_{ik}\) and projected so that
\(\sum_i\Delta p_i=0\). In rotationally invariant, force-only settings it must also
satisfy \(\sum_i(x_i-x_{cm})\times\Delta p_i=0\), or be represented by central pair
forces \(J_{ij}=\alpha_{ij}(x_i-x_j)\) with symmetric scalar \(\alpha_{ij}\).

Use a stated integrator. For conservative unconstrained systems, use a symplectic or
variational update; for fluids with viscosity, walls, forcing or drag, state exactly
which quantities should *not* be conserved and apply boundary/forcing terms separately.
Do not promise Hamiltonian or energy conservation in dissipative data.

### Required propositions and tests

| Proposition | Proof obligation | Automated verification |
|---|---|---|
| Permutation equivariance | Show restriction, sparse coarse graph construction, pair updates and prolongation commute with any particle permutation. | Random permutation equality to tolerance, including variable masks. |
| E(2)/E(3) equivariance | Show scalar gates depend only on invariants and vector/tensor operations transform under the selected group. | Translation, rotation and reflection property tests over random scenes and boundary cases. |
| Linear momentum conservation | Antisymmetric pair impulses and no unmodelled external force imply \(\sum_i\Delta p_i=0\). | 10k randomized force-free scenes; report max and quantiles of violation. |
| Angular momentum conservation, if claimed | Require central forces or a torque-free projection; prove it. | Same test around multiple origins. |
| Partition of unity | \(w_{ik}\ge0\), \(\sum_k w_{ik}=1\), no empty/zero-mass group. | Unit tests, adversarial clustered scenes and masked-particle tests. |
| Scale consistency | Define the physical restriction \(R\), coarse update \(F_c\), fine update \(F_f\), and quantify \(\|R F_f-F_cR\|\). | Manufactured solutions and resolution-refinement convergence. |
| Complexity | Derive and measure edge counts and peak memory: local \(O(Nk)\), coarse \(O(Kk_c)\), restriction/prolongation \(O(Nq)\). | Profiling from 10^2 through the largest feasible N; no dense hidden tensors. |

## Complete improvement checklist

### P0 — invalidate or replace the current core

- [ ] **Stop treating the legacy autoencoder as the candidate simulator.** Its primary
  outcome is reconstruction, whereas the research question is predictive dynamics.
  Build RIPII-MR as a direct state-transition model; keep autoencoding as an optional
  representation baseline only. Verify that every headline metric is a rollout metric.
- [ ] **Delete “renormalization” from public claims and variable names where feasible.**
  The current \(BB^\top\) operator is a learned projection, not a renormalization-group
  transformation. A future RG claim requires an explicit scale map, flow/composition law,
  fixed-point analysis and empirical scale-transfer evidence.
- [ ] **Remove VQ, latent action, fusion gate and all unsupported auxiliary losses from
  the default path.** Each must earn re-entry through an individually frozen, compute-
  matched ablation. Verify that `core` has no inactive loss parameters or hidden routes.
- [ ] **Replace raw-frame assignment \(\operatorname{softmax}(W h_i)\).** It does not
  guarantee geometric locality, stable identity, E(2)/E(3) equivariance, mass-aware
  aggregation, or a meaningful scale. Implement compact-support geometric weights and
  compare against fixed spatial bins, FPS/Voronoi grouping and no coarse path.
- [ ] **Remove fixed correction multipliers.** Predict acceleration or impulse in
  normalized units, document normalization constants, and integrate with explicit
  \(\Delta t\). Verify unit changes transform predictions correctly and train/test use
  identical normalization fitted only on training data.
- [ ] **Make physical assumptions explicit.** For each dataset record whether particles
  have mass, material, boundaries, force, viscosity and valid neighborhood topology.
  Disable conservation claims when walls, drag or external forcing are present.

### P0 — make the new architecture falsifiable

- [ ] **Implement a minimal local equivariant baseline first.** It must predict impulses
  or accelerations, use relative geometry, respect masks and expose identical state and
  integrator interfaces. This is both a baseline and the fine processor of RIPII-MR.
- [ ] **Implement conservative interaction decoding.** Use antisymmetric directed-edge
  pairing or central pair forces; never rely on a penalty to “usually” conserve momentum.
  Verify exact force-free conservation and controlled changes under known forcing.
- [ ] **Implement restriction/prolongation with a partition of unity.** Include mass or
  volume weights, sparse memberships, nonempty-group handling and a deterministic anchor
  policy. Verify no arbitrary group-label dependence and no dense \(N\times K\) path at
  target scale.
- [ ] **Make coarse interactions sparse and topology-aware.** Use graph/mesh adjacency,
  coarse radius or kNN; benchmark it against dense global pooling at equal budget.
- [ ] **Define adaptive routing before any results.** The gate can use a local residual
  uncertainty or graph-diameter/receptive-field statistic, but it must be causal and
  thresholded before test inspection. Report activation rate and compute saved; an
  always-on hierarchy is not adaptive.
- [ ] **Add a scale-consistency loss only after the base trains.** It must compare
  physically restricted predictions, not arbitrary latent vectors. Sweep its one weight
  on development only; include zero-weight and random-coarsening controls.

### P1 — training, optimization and numerical analysis

- [ ] **Use staged objectives:** (1) one-step acceleration/impulse loss; (2) scheduled
  multi-step rollout loss; (3) optional physically meaningful constraint terms. Add no
  learned uncertainty weights until an ablation proves they help. Log every raw term,
  gradient norm/cosine and effective coefficient.
- [ ] **Train for rollout stability.** Calibrate state noise from each dataset, use
  pushforward/scheduled rollout horizons, and compare teacher forcing, noise only,
  rollout only and the combined schedule. GNS identifies message-passing depth and
  training-noise corruption as major long-horizon levers.^1
- [ ] **Perform optimizer and integrator studies.** Sweep learning rate, weight decay,
  processor depth, local radius, coarse spacing, coarse depth, gate threshold and
  rollout horizon under a fixed development budget. Check finite differences for custom
  projections and run mixed-precision/nonfinite stress tests.
- [ ] **Establish numerical baselines.** Compare Euler, semi-implicit Euler and the
  applicable symplectic/constraint solver at equal function evaluations. A learned
  architecture cannot claim stability if the integrator difference explains it.
- [ ] **Prevent capacity and tuning confounds.** Match parameter count, processor depth,
  hyperparameter trials, updates, tokens/particle steps, wall time and hardware. Report
  a Pareto frontier rather than a single cherry-picked width.
- [ ] **Support variable N from day one.** No silent truncation/padding as a scientific
  result. Test masks, empty neighborhoods, nonuniform density, domain-size change and
  N larger than training.

### P1 — mechanism evidence

- [ ] **Create manufactured long-range tasks.** Examples: harmonic all-pairs coupling,
  incompressible pressure-like global response, separated-body constraint transmission,
  and a matched purely local task. Derive exact or high-accuracy reference behavior.
- [ ] **Measure why hierarchy helps.** Report receptive-field coverage, coarse gate rate,
  assignment compactness, mass conservation, restriction/prolongation error, spectral
  error by spatial wavelength and error-versus-distance. Assignment entropy alone is not
  mechanism evidence.
- [ ] **Use decisive controls.** Compare: local graph; local graph with larger radius;
  global mean/attention; fixed geometric hierarchy; random hierarchy; nonconservative
  hierarchy; conservative hierarchy without adaptivity; and full RIPII-MR. This isolates
  hierarchy, geometry, conservation and routing separately.
- [ ] **Test counterfactual intervention.** Disable the coarse path only on gate-positive
  scenes at inference; shuffle coarse messages between matched scenes; change coarse
  resolution. A real mechanism should fail specifically under these interventions.
- [ ] **Measure calibration/uncertainty only if used for routing.** Report reliability,
  selective-risk curves and whether gate activation predicts local-baseline error. Never
  use uncertainty as a decorative auxiliary output.

### P1 — benchmark and experiment design

- [ ] **Adopt a benchmark ladder.**

  | Tier | Role | Required evidence |
  |---|---|---|
  | Analytic/manufactured | proves algebra and isolates long range | exact errors, conservation, scale tests |
  | Current small object world | rapid falsification and failure figures | local vs global vs hierarchy boundary |
  | LagrangeBench 2D/3D | public particle-fluid comparison | complete untruncated task adapter and official split protocol |
  | Mesh/continuum public tasks | resolution/geometry scaling | compatible MeshGraphNets/MS-MGN/Bi-Stride comparison |
  | Independent held-out domain | external generalization | frozen protocol, untouched test, retained failures |

  LagrangeBench provides public Lagrangian fluid data, loading/preprocessing utilities,
  official GNS/EGNN/SEGNN-type baselines and rollout metrics including position MSE,
  kinetic energy and Sinkhorn distance.^2 It is appropriate only after RIPII can handle
  the actual variable particle count; the present 2,708-particle LDC task must not be
  truncated to the internal rigid-disc interface.

- [ ] **Run the correct baselines.** At minimum: constant velocity/known-force numerical
  baseline; local GNS; EGNN or SEGNN; global-context control; Neural SPH-style physical
  augmentation where data support it; constraint-based GNS; fixed multiscale method;
  and an official/faithful MS-MGN or Bi-Stride comparison on compatible mesh data. Do
  not compare incompatible input representations as though they were fair.
- [ ] **Evaluate strong physics metrics.** Position/velocity error by horizon; energy
  where meaningful; momentum/torque residual where applicable; density/pressure or
  divergence for fluids; wall penetration/contact; Sinkhorn/distributional error;
  stability failure rate; inference latency, training time, memory and FLOPs.
- [ ] **Define generalization before training.** Hold out initial conditions, N,
  resolution, physical parameters, forcing amplitudes, boundary geometries and long
  horizons. Use at least one interpolation and one extrapolation regime.
- [ ] **Use independent data seeds.** Model seeds alone are not replication. Preserve
  per-trajectory metrics, test predictions and every failed/nonfinite run.
- [ ] **Power from a blinded development pilot.** The unit is a trajectory/scene paired
  across models, nested within data generation. Set the minimum number of independent
  scenes and training seeds to detect the predeclared practical effect; five paired seeds
  cannot yield a two-sided exact sign-flip p below 0.0625.
- [ ] **Freeze a two-stage plan.** Development selects architecture once. A hash-bound
  confirmatory protocol then fixes models, trials, budgets, endpoints, exclusions and
  multiplicity family before any untouched-test evaluation.

### P1 — ablation matrix

| Question | Required ablation | Pass condition |
|---|---|---|
| Does hierarchy matter? | local equivariant vs full, same capacity/compute | full improves predeclared long-range endpoint |
| Is geometry essential? | learned geometric vs fixed bins/FPS vs random assignments | only geometry-aware grouping gives benefit |
| Is conservation causal? | conservative decoder vs same decoder without projection | better long rollout without trading away accuracy |
| Is routing useful? | adaptive vs always-on vs always-off at matched max budget | adaptive lies on a better accuracy--compute frontier |
| Is scale consistency useful? | zero, selected, high weight; random restriction control | selected weight improves held-out scale transfer |
| Is the effect just larger context? | local radius/depth sweep and global pooling/attention | full wins equal-receptive-field controls |
| Does it scale? | N and resolution sweep, fixed physical domain | error/compute scaling beats flat baseline |
| Does it transfer? | parameters, boundaries, forcing and simulator/domain holdouts | effect persists where hypothesis predicts |
| Does it survive training? | seed/data-seed/optimizer/integrator robustness | direction and interval meet frozen criterion |

### P2 — dataset quality and provenance

- [ ] **Write a dataset contract per benchmark:** source DOI/revision/license/hash,
  state units, coordinate system, time step, particle semantics, boundary encoding,
  train/validation/test allocation, preprocessing fit scope and known defects.
- [ ] **Preserve physical units and nondimensionalize consciously.** Store both raw and
  normalized values; document reference length, mass, time and velocity scales; test
  whether unit rescaling changes learned behavior appropriately.
- [ ] **Audit simulator-to-model mismatch.** LagrangeBench is SPH fluid data; current
  RIPII assumes small rigid-disc objects. Build an adapter only if it represents fluid
  particles, boundaries and histories faithfully, otherwise use a different benchmark.
- [ ] **Add domains that distinguish the claimed mechanism.** Use local and deliberately
  nonlocal physical regimes, 2D and 3D where the method claims both, and mesh/particle
  cases only when one representation can be handled without lossy conversion.
- [ ] **Treat real-world data cautiously.** Do not call motion-capture or generic tracked
  trajectories “physical validation” unless state estimation, controls, units, contact
  and ground truth support the exact claim. If adding experimental data, include sensor
  noise, missing tracks, calibration uncertainty and a solver/measurement baseline.
- [ ] **Check leakage and contamination.** Hash trajectories, group splits by simulated
  scene/initial condition, prevent temporal-window overlap across splits, and maintain a
  dataset-version registry. Never tune on public test leaderboards repeatedly.

### P2 — implementation and reproducibility

- [ ] **Create a new package boundary** (`ripii_mr/` or equivalent) rather than silently
  mutating legacy RIPII. The old code/results remain reproducible negative evidence.
- [ ] **Define typed state and unit contracts.** Separate particle state, boundary state,
  material parameters, controls, graph construction and integrator. Reject missing or
  semantically incompatible fields.
- [ ] **Add reference implementations.** A slow, transparent NumPy/PyTorch version of
  restriction, conservation projection and one rollout should be compared with the fast
  kernel on randomized cases.
- [ ] **Add invariant/property-based tests.** Include permutation, Euclidean transforms,
  conservation, mask invariance, finite gradients, empty neighborhoods, variable N,
  checkpoint-resume equality and deterministic CPU replay.
- [ ] **Profile honestly.** Measure end-to-end graph construction, memory, data transfer,
  training and inference--not only neural-layer FLOPs. Publish hardware, precision,
  compiler settings and warmup policy.
- [ ] **Release complete artifacts.** Frozen configs, source revision, dependency lock,
  raw per-seed outputs, checkpoints where licenses allow, metrics code, plots/tables
  generated from artifacts, an immutable archive and a clean-machine reproduction log.

### P3 — high-value extensions after the core passes

- [ ] **Learn a physically meaningful adaptive mesh/anchor policy.** The policy may
  allocate anchors using estimated local truncation error, density or vorticity, but must
  retain partition-of-unity, symmetry and compute constraints.
- [ ] **Operator-learning extension.** Learn a parameter-conditioned interaction operator
  across viscosity, forcing and geometry while testing true extrapolation; do not call
  this foundation-model generalization without broad evidence.
- [ ] **Constraint-solver hybrid.** Use RIPII-MR as a proposal followed by a fixed number
  of constraint/projection iterations. C-GNS demonstrates that learned constraints plus
  iterative inference can improve simulation and support new constraints at test time.^3
  Compare equal wall time and solver iterations.
- [ ] **Differentiable inverse/control study.** Only after forward accuracy is proven,
  evaluate gradients against a differentiable solver or finite differences for parameter
  inference/control. This is a new claim with its own baselines.
- [ ] **Probabilistic closure/residual model.** For unresolved subgrid dynamics, model a
  calibrated stochastic residual and evaluate distributional rollout metrics, not only
  a point estimate.
- [ ] **3D irreducible-representation version.** If a 2D EGNN-style method succeeds,
  compare against SEGNN/e3nn-style models; do not extrapolate 2D symmetry results to 3D.

## Suggested execution order

1. Freeze the new claim, remove legacy terms from the default, and implement the local
   equivariant conservative simulator with theorem/property tests.
2. Implement fixed geometric multiresolution restriction/prolongation and conservative
   sparse coarse correction. Run manufactured tasks and all decisive controls.
3. Add adaptive routing and scale consistency only if fixed multiresolution has a clear
   mechanism signal.
4. Build a task-valid LagrangeBench adapter and reproduce selected official baselines on
   one development case before designing the full matrix.
5. Run a compute-matched development benchmark ladder, choose exactly one configuration,
   then freeze the confirmatory protocol.
6. Run the independent confirmation, release raw evidence, and write only claims that
   survive it.

## Implementation status (2026-09-14)

Implemented and verified for this repository:

- [x] Separate direct state-transition candidate `ripii_mr`; legacy checkpoints and
  negative evidence remain unchanged.
- [x] Scalar-invariant encoding, central antisymmetric fine/coarse impulses and an
  explicit semi-implicit force/impulse update without the legacy position bypass.
- [x] Mass-aware geometric restriction, sparse memberships, partition of unity, sparse
  coarse kNN edges, mass-share prolongation and closed-form torque projection.
- [x] Deterministic hard evaluation routing with a straight-through training gate and
  mechanism diagnostics.
- [x] Tests for E(2), permutation behavior on generic scenes, momentum/torque,
  restriction mass, partition of unity, force semantics and gradient reachability.
- [x] End-to-end integration with training, checkpointing, evaluation, capacity matching,
  manifests and CLI model selection.
- [x] Frozen three-data-seed/five-model-seed development protocol and resumable outer
  matrix runner with a fail-closed advancement rule.
- [x] Prospective methods manuscript and verified PDF build.

Still scientifically unresolved and therefore intentionally unchecked: the full 75-run
development result, decisive component ablations, external LagrangeBench/mesh adapters,
subquadratic neighbor construction, equal-FLOP/wall-time tuning, powered confirmation,
and independent reproduction.

## Stop/go criteria

| Gate | Go | Stop or narrow |
|---|---|---|
| Algebra | all claimed properties pass to numerical tolerance | remove failed physical claim; do not compensate with a loss |
| Mechanism | hierarchy beats local/global/fixed/random controls only in induced long-range regimes | call hierarchy unnecessary and preserve the negative result |
| Scaling | better accuracy--compute Pareto curve at increasing N/resolution | do not claim scalability |
| External benchmark | predeclared primary endpoint, uncertainty and physical diagnostics pass | no superiority claim; report benchmark-specific result |
| Reproduction | clean independent replay agrees from released artifacts | do not claim reproducibility/ready-for-use |

## Literature position

Graph Network-based Simulators made particle-state message passing and noise-trained
rollout simulation a central learned-simulation baseline.^1 MeshGraphNets established
graph learning for mesh simulation and adaptive resolution; MultiScale MeshGraphNets
already demonstrates fine/coarse message passing for accuracy and efficiency at high
resolution.^4,5 E(n)-equivariant GNNs provide an established route to exact Euclidean
and permutation equivariance, while fluid work in LagrangeBench shows that equivariant
models can improve learned particle interactions but may cost more to train/evaluate.^6,7
Neural SPH shows a different lesson: carefully chosen solver-derived pressure, viscous
and external-force structure can improve long rollouts by orders of magnitude.^8

Therefore the relevant standard is not “does a soft hierarchy work?” It is whether a
new conservative, adaptive, geometry-aware hierarchy improves the **measured
accuracy--cost tradeoff** beyond established multiscale, equivariant and physics-guided
alternatives.

## Sources

1. Sanchez-Gonzalez et al. “[Learning to Simulate Complex Physics with Graph Networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html).” ICML, 2020.
2. Toshev et al. “[LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite](https://papers.neurips.cc/paper_files/paper/2023/hash/ccac3b120c7dc86d45f56830732b62be-Abstract-Datasets_and_Benchmarks.html).” NeurIPS Datasets and Benchmarks, 2023; [documentation](https://lagrangebench.readthedocs.io/en/stable/).
3. Rubanova et al. “[Constraint-based graph network simulator](https://proceedings.mlr.press/v162/rubanova22a.html).” ICML, 2022.
4. Pfaff et al. “[Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409).” ICLR, 2021.
5. Fortunato et al. “[MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612).” 2022.
6. Satorras, Hoogeboom and Welling. “[E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).” ICML, 2021.
7. Toshev et al. “[Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks](https://arxiv.org/abs/2305.15603).” 2023.
8. Toshev et al. “[Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics](https://arxiv.org/abs/2402.06275).” ICML, 2024.
9. Cao et al. “[Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network](https://proceedings.mlr.press/v202/cao23a/cao23a.pdf).” ICML, 2023.
