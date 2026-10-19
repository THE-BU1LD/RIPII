# Closure study v1: prospective scientific specification

**Written before implementation or outcome-bearing training, 2026-10-19.**
This is a separately versioned successor to the October 19 contract/mechanics
audit. It does not redefine `ripii_mr`, `ripii_mr_v2`, their checkpoints, the
historical development studies, or their negative conclusions. The machine
protocol in `protocol.json` supplies the exact experimental constants.

## 1. Question, motivation, and contribution boundary

An internal conservative predictor can be incompatible with an open system when
its external impulse omits drag or boundary reactions. A different observation
integrator adds an independent discrepancy. The retained October 19 diagnostic
demonstrates both obstructions on literal single-disc fixtures, without training.
It does not identify how much of a missing historical training run's error they
caused. Closely related external/internal decompositions, equivariant graph
hierarchies, and integrator-aware learning already exist; see `RELATED_WORK.md`.

**Precise direction:** action-only constrained predictors struggle with open
disc dynamics under drag, wall contact, and mismatched substeps. We test shared
external closure and consistent kick-drift updates because force balance predicts
that these remove the demonstrated single-disc obstruction. We then test whether
a hierarchy supplies any additional predictive value against equally informed,
exact-parameter and compute-budget controls.

The defensible contribution is a controlled, reproducible representability and
mechanism diagnostic, including a possible negative hierarchy finding. We do not
claim a new force-decomposition principle, an official GNS/EGNN/EGHN reproduction,
large-system acceleration, or state-of-the-art performance. A known-law solver
is an oracle reference. An accurate oracle proves parity, not learned novelty.

## 2. Hypotheses and falsifiers

**H1, correctness:** external closure plus four substeps reproduces a single
disc's known dynamics to floating-point tolerance; one-step and action-only
ablations retain the analytically predicted discrepancy. Failure on those cases
invalidates the implementation. Learned rollout improvement is a separate,
descriptive secondary comparison, not a consequence of this theorem.

**H2, hierarchy (primary empirical hypothesis):** after sharing closure and the
four-substep integrator, the hierarchy improves mean OOD position RMSE by at least
5% against the same-parameter flat all-pair control when that control receives
at least the hierarchy's measured optimization time, within the update cap.
The claim also requires no >5% IID regression and ≥5% mean OOD benefit over both
the learned-mean control and one-parameter physical-law control. These additional
controls are mandatory falsifiers: a simpler successful explanation rejects a
broad hierarchy-advantage claim. No guarantee of successful optimization follows
from conservation, equivariance, or a representability proof.

## 3. State, information, units, and assumptions

State is `s[B,N,6]=(x,y,vx,vy,r,m)`, force action is `a[B,N,2]`, and boolean mask
is `mask[B,N]`. Output has the same state shape. Each scene has 1–8 live discs at
inference; study training draws 2–4 and count OOD draws 5–8. State and action are
finite floating tensors with identical dtype/device; live radius/mass are
positive. Padding is sanitized before arithmetic and returns exact zero.
Experiments use dimensionless simulation units, float32, radii .07/.11 and masses
.7/1.4. Mathematical tests use float64. The new namespace accepts float32/float64
and rejects lower precision because its specified geometry floor is not safe in
float16. No physical live mass is clamped in the
successor. Simulator parity is claimed only for masses ≥.1, because the existing
simulator clamps smaller masses and preserves a different contract there.

All arms receive identical state, action, live count and the fixed known
environment: square walls `[-1,1]^2`, wall stiffness 100, wall damping 1.5,
drag .06, and observation interval .05. No varying hidden environment parameter
is tested. The generator's global coupling is fixed at 1; learned models are not
given that value. The physical-law and mean controls are told its functional
form and estimate its coefficient from the same training trajectories.
Known wall and contact coefficients coincide in this generator. The physical-law
control explicitly exploits this information; it is a strong, more structured
prior, not an equally flexible neural architecture. Every arm's information and
inductive bias are disclosed. The oracle is told the entire generator law.

Inputs are fully observed and Markov for fixed environment. Different unknown
drag/arena parameters cannot generally be identified from one identical state
and action; no such generalization claim is made. The fitted coefficient is a
discrete-surrogate estimate under the stated solver, not a proof of universal
continuous-law discovery.

## 4. External force and observation update

At every substep `h=dt/S`, recompute all forces at the current position/velocity.
For each outward wall normal `n` with plane `n·x=1`, penetration is
`δ=n·x+r−1`. Its force is `−n max(kδ+c n·v,0) 1[δ>0]`.
External closure is

`F_ext = a − γv + Σ_walls F_wall`.

For closure-off cells `F_ext=a`; all other environment information remains
available to the model, but the declared ablation removes these exact terms.
The internal model produces **forces**, not observation-sized learned impulses.
For each of S substeps, execute

`v_next = v + h (F_internal + F_ext)/m`,
`x_next = x + h v_next`.

Radius and mass remain unchanged. Actions are held constant across the
observation's substeps, as in the generator. No post-step collision clamp,
projection onto the arena, adaptive step count, router, or extra learned
position decoder is used.

## 5. Neural internal force, chronological specification

### 5.1 Scalar node encoder

At each substep form six invariants per particle:
`[|v|², |a|², v·a, radius, mass, live_count/8]`.
An MLP `6→16→16→16`, with SiLU after its first two linear layers, produces
`z[B,N,16]`. Mask padding. Weights/biases use PyTorch Linear defaults, with each
arm initialized from the same paired seed; the encoder and fine head therefore
start identically. No quantizer, normalization, dropout, or regularization is used.

### 5.2 Symmetric central pair head

For each undirected edge `i<j`, let `d=x_i−x_j`, `u=v_i−v_j` and
`d2=max(|d|²,1e−12)`. Input is
`[z_i+z_j, z_i*z_j, d2, |u|², d·u]` (35 scalars).
An MLP `35→16→16→1`, SiLU after the first two layers, predicts
`q=20 tanh(MLP(input))`. Its final weight and bias initialize to zero.
The pair adds `q d/sqrt(d2)` to i and its negative to j. Coincident particles
receive zero directional pair force; this is finite but not a learned collision
resolution guarantee. The fine head uses all live pairs at distance <.6.

### 5.3 Flat global control

A second independently initialized head of exactly the same shape operates on
all live pairs. Add its force to the fine force. The two-head flat control and
hierarchy have the exact same trainable shapes and parameter count; live count
is supplied to both so the `1/N` coupling normalization is representable.
All-pair processing is O(B N² H²) at fixed MLP depth. It is deliberately strong
at N≤8 and has no coarsening overhead. No sparse large-N claim follows.

### 5.4 Hierarchy

Use four geometric anchor slots and two memberships per particle. Choose the
first anchor farthest from the unweighted live center, then successive anchors
by farthest-point sampling of nearest-anchor distance; active anchor count is
`min(4,live_count)`. For each particle retain its two nearest active anchors
(or the only anchor for one live particle). Weight them with a masked softmax
of `−distance²/(2 scale²)`, where `scale²` is mean squared distance to the live
center, floored at `1e−4`. Padding softmax rows must remain finite before masking.

For weights `w_ik`, define actual coarse mass `M_k=Σ_i w_ik m_i`; an active
coarse node has `M_k>0`, and shares are `p_ik=w_ik m_i/M_k` on active nodes,
zero otherwise. Use safe conditional denominators, not clamped physical masses.
Restrict features, position and velocity with p. Apply the second central head
to every active coarse pair, and prolong its force by `F_i=Σ_k p_ik F_k`.
This preserves total force in real arithmetic. Remove residual translational
and rotational wrench as specified next; add the fine force. No trainable
router is present, so coarse computation is always charged.

Farthest-point and nearest-anchor exact ties are index-sensitive. Conservation
and finite-output claims cover these cases; arbitrary permutation equivariance
is claimed only for nondegenerate selections with unchanged tie ordering, and
numerically checked on generic fixtures. Do not claim a universal tie theorem.

### 5.5 Wrench projection

For live total mass M, center c and r_i=x_i−c, first set
`F_i ← F_i − (m_i/M) Σ_j F_j`. Then compute
`τ=Σ_i cross(r_i,F_i)` and `I=Σ_i m_i |r_i|²`.
For I>0, set `F_i ← F_i − (τ/I) m_i (-r_iy,r_ix)`;
for I=0 use zero coefficient. Safe conditional denominators must avoid NaN
gradients. This is a rigid-wrench correction and adds no learned parameters.
Apply it to the prolonged force and total hierarchical internal force.

### 5.6 Simpler controls

`mean`: the same fine encoder/head plus `−softplus(ρ)(x_i−mean_live(x))`.
Initialize the learned coupling at .25. This has one extra parameter and the
exact task-appropriate mean-field form; it requires O(BN) global work.

`law`: use the simulator's known central soft-contact law with k=100,c=1.5,
and the same one learned nonnegative mean-field coefficient initialized at .25.
Contact force for an overlapping pair is
`max(k(r_i+r_j−d)−c(v_i−v_j)·n_ij,0) n_ij`.
This arm has only one trainable scalar. Its stronger physical prior is explicit.
`prior`: zero internal force, shared external closure and four substeps.
`oracle`: unchanged full simulator with true coupling 1. Oracle results are
sanity references and are excluded from learned-winner significance testing.

## 6. What the construction proves

Pair antisymmetry gives zero summed internal force; central direction gives
zero pair torque. Partition-of-unity mass shares preserve summed coarse force.
The projection removes net force and torque in real arithmetic under positive
live mass and safe inactive-group handling. During a kick,
`P_next−P=h Σ F_ext` and `L_next−L=h Σ x×F_ext`.
The subsequent drift adds no angular momentum because each `v_next×v_next=0`.
Iterating these identities proves the appropriate external-force balance across
substeps. Floating-point residuals, nonsmooth selections, and finite precision
remain measurable numerical issues; global long-horizon stability is not proved.

For constant one-disc acceleration a, S semi-implicit substeps give
`v'=v+dt*a`,
`x'=x+dt*v+((S+1)/(2S))*dt²*a`.
At S=4 the acceleration coefficient is 5/8, compared with 1 at S=1.
For force-free drag without walls, `v'= (1−hγ/m)^S v`; position is the
finite geometric sum of intermediate velocities. With one particle, the neural
internal pair terms vanish; the corrected model must match the simulator.

Central/internal operators are E(2)-equivariant under generic geometric
selections. The complete fixed-square system only has the square's valid
symmetries (including quarter-turns/reflections). Arbitrary particle translation
with fixed walls is not a symmetry. No full E(2) external-boundary claim is made.

## 7. Training and inference

All neural arms use Adam lr .003, betas (.9,.999), epsilon 1e−8, no weight decay;
the one-scalar law control uses lr .03. Train four-observation autoregressive
windows, batch eight, uniformly sampled scene/start with one shared generator
sequence per paired unit. Randomly permute all slots in each window. Minimize
the mean over four steps of the existing masked squared-state loss with
coordinate weights `[4,4,1,1]` and denominator ten times live object count.
No teacher forcing inside the four-step window, gradient truncation within it,
noise augmentation, auxiliary loss, scheduler, or development tuning is used.
Clip global gradient norm at 5; retain losses and preclip norms. Reject nonfinite
loss, gradients or output, record the failed run, and do not substitute seeds.

Validate position RMSE over a 32-step free rollout after each 40 updates and at
the final update. Select the lowest validation position RMSE (first checkpoint
on exact tie). Selection uses only validation; every model is evaluated at that
selected checkpoint. Save selected and final states, optimizer and window RNG
state, initialization/source/config hashes, all losses and validation scores.
The flat compute control continues the same optimizer/window stream after the
base update budget until measured optimization time reaches the hierarchy's,
or four times the base updates. Its validation candidate set includes the base
checkpoints. Record any cap-limited failure to cover the target compute.

Inference repeats the same forward update, returns exact unchanged properties
and zero padding, has no training-only routing behavior, and loads only strict
`ripii-closure-v1` checkpoints. Malformed schema/config/weights must fail rather
than reinterpret old WorldModel checkpoints. Source/model version is explicit.

## 8. Methodology and finite budget

Three independent development data/model-seed pairs and six fresh confirmation
pairs are fixed in the protocol. Training data: 96 scenes ×16 observations;
validation:16×32; evaluation:24×32 in IID, more_objects, composition and fast.
All arms in a pair share exact tensors/windows; independent unit is the paired
data/model-seed draw, not timestep or particle. OOD scenes are separate from
validation and training. Save tensors/content hashes, full generation specs,
IDs and source offsets; verify no within-split duplicate initial-state hashes
and no cross-split overlap. Data are generated from the repository's synthetic
source. Its license metadata is `NOASSERTION`: `LICENSE_STATUS.md` records no
owner-approved repository license. Do not borrow the separate NRI source's MIT
license or claim redistribution terms have been resolved. No external/private
dataset is included.

Primary error is masked position RMSE over all 32 future observations, normalized
by live coordinates/time within each regime; average the three OOD regime RMSEs
equally. Do not pool particles across regimes. Retain per-scene squared errors,
position/velocity RMSE, endpoints at 1/4/16/32, scene momentum error, contact and
wall penetration, and maximum state magnitude. Initial position/velocity noise
σ=.01 on the same IID scenes is a prespecified descriptive robustness diagnostic;
targets remain clean, so this measures observation-noise sensitivity. Record
accuracy by live-count bin and horizon, and timing at N=2,4,8. No unseen force-law,
unknown-wall, timestep, or high-particle-count claim is tested.

The base update budget is 120, lowered only to 80 if a timing-only dry run
projects the full grid above 1800 seconds; if even 80 projects above 1800, stop
before science as budget-inconclusive. Dry-run cap180 seconds; outcome-bearing
wall cap2100 seconds. Runtime projection uses measured full pipeline cost, a
1.3 safety factor, and no accuracy-based choice. There is one process/one Torch
thread at a time. No outcomes-driven scientific revisions are permitted in v1.
Budget or finite-output failures retain the complete partial grid and produce
inconclusive/invalid dispositions according to cause, never positive extrapolation.

After valid development, freeze exact source commit, protocol, environment,
architecture, metrics, hyperparameters, seeds and analysis before generating any
confirmation tensors. Confirmation is not conditional on favorable development
results: a scientifically valid negative test is eligible. No retuning, optional
subgroup selection, seed replacement or altered endpoints after inspection.

## 9. Statistics, decisions, and reporting

For each independent pair compute `log(RMSE_hierarchy/RMSE_global_compute)`.
Report all values, arithmetic regime RMSE means, geometric error ratio, and
a paired-unit percentile bootstrap 95% interval (10000 draws, fixed seed46001).
With six units this interval is descriptive and may have poor coverage.
Also enumerate all 2^6 sign flips of the paired log ratios around log(.95),
using a two-sided mean statistic; this test relies on symmetry/exchangeability
of paired errors and is not a randomized-treatment experiment. There is one
primary test. All other uncertainty, subgroup and secondary results are
descriptive; no multiplicity-adjusted broad discovery claims are made.

H2 is SUPPORTED only when the geometric ratio and upper interval bound are <.95,
the primary sign-flip p≤.05 and compute coverage is complete. The IID safeguard
is the across-unit geometric hierarchy/global_compute IID ratio ≤1.05; it is
not a requirement on every individual seed or its interval. Each simpler-control
safeguard is the across-unit geometric OOD hierarchy/control point ratio <.95;
only the primary control additionally requires its interval and sign-flip gate.
If a required control rules out ≥5% benefit (its lower
ratio interval is ≥.95), H2 is NEGATIVE within this budget/distribution. Otherwise
H2 is INCONCLUSIVE, with descriptive MIXED behavior recorded where appropriate.
Correctness H1 and empirical H2 are classified separately. Incomplete compute
coverage forces INCONCLUSIVE for the overall H2 decision; comparisons against
other controls remain descriptive. Missing grid units,
leakage or an implementation defect cannot count as a negative scientific test.

Every important number must trace to protocol→data/source→checkpoint→predictions
→analysis→table/figure. Regenerate metrics from saved predictions independently,
replay selected checkpoint rollouts, verify exact artifact set/hashes and split
identities, and retain all failures. The final manuscript must distinguish
theorem, numerical check, development, protected confirmation and limitation.
Prepare code, evidence, figures, bibliography and PDF; publication/submission
and default-branch merge remain outside this task's authorization.

## 10. Prospective gate record

IDEA READY: yes for a bounded diagnostic contribution, not for an architecture
novelty claim. MODEL SPECIFICATION READY: yes subject to implementation audit of
these equations. METHODOLOGY READY: the protocol is prospective; final code and
budget timing record must be frozen before protected confirmation. Later gates
remain pending and are recorded in the canonical state. No P0–P6 readiness
labels are reassigned by this document.
