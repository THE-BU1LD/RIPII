# Closure study v1: prospective adversarial review

Date: 2026-10-19.

Status: **prospective diagnostic design review; no successor outcome is asserted**.
This document and [FLAWS.csv](FLAWS.csv) do not assign or change a repository
readiness score. The register is a review of identifiable flaws and alternative
explanations, not a promise that every possible defect has been discovered.

## Question and permitted contribution

The successor separates two questions:

1. In a local central-force model, what changes when known external closure is
   off versus on and integration uses one versus four substeps?
2. With the corrected external/integration contract shared, does an always-on
   hierarchy improve the prespecified rollout endpoint over equally informed
   full-pair/global and learned analytic-mean/simple parametric controls?

These are diagnostic questions. Repairing a transition contract is scientifically
useful if it localizes a failure, but it is not an architecture novelty claim.
A closure effect cannot by itself establish hierarchy utility. A hierarchy result
cannot retroactively verify the missing predecessor run.

The scope is the coupled, two-dimensional synthetic object world, with at most
eight live objects, masses at least 0.7, fixed observation step 0.05, the fixed
arena, and known wall/drag environment shared across competitive learners.
Only the protocol's declared synthetic shifts and separately protected seeds
belong to the evaluation. There is no router, arbitrary geometry or observation-dt
shift, large-particle benchmark, perception task, or general law-identification
claim in this study.

The study protocol must supply the exact architectures, seed registry,
thresholds, budgets, horizons, fitting procedures, and decision rules. This review
does not invent additional trials or override that protocol. Historical model
names, checkpoints, frozen protocols, and negative results retain their meanings.

## What the predecessor diagnosis establishes

The retained [October 19 mechanics report](../results/development/ripii_contract_audit_20261019/report.md)
demonstrates action-only conservation and integration obstructions on explicit
synthetic fixtures. The [simulator](../../ripii/world/physics.py) includes wall
forces and drag, holds action constant within an observation, and advances through
four semi-implicit substeps. A single object cannot use internal pair forces to
represent a missing external force. Under constant force, one semi-implicit step
also has a different position increment from four substeps.

This is a reason to test the new contract. It is not evidence that a trained
successor will win. The earlier aggregate MR errors cannot be decomposed using
these fixtures, because their original trajectories and checkpoints remain
unavailable. The old action-only lower bound must not be applied to a corrected
transition whose external forces change with its evolving predicted state.

The current evidence and archive limitations are qualified in the
[evidence ledger](../../EVIDENCE_LEDGER.md) and
[engineering handoff](../../docs/SESSION_HANDOFF_2026-10-19.md).
Their original negative outcomes remain part of the new study's lineage.

## The strongest alternative explanations

### The change only supplies omitted physics

Closure-on must use its own current predicted position and velocity. Supplying
future reference positions, reference velocities, or hidden simulator substeps
would turn the closure into target leakage. Known coefficients and the boundary
contract must be explicit for every model.

The first-stage factorial holds the local central-force family fixed while
crossing external closure off/on with one/four substeps. The intended contrasts
must not also change width, data, loss, noise, checkpoint selection, or tuning
allowance without disclosure. Internal learned forces must be scaled by the
actual substep duration. Four substeps also do more numerical work, so the
factorial estimates the declared numerical intervention, including that cost.

Single-object and known-force sanity cases can establish implementation parity.
They cannot establish a learned advantage. A zero-learned-residual or prior-only
control distinguishes what the supplied physics accomplishes by itself.

### A simple mean explains the nonlocal force

For the generator's harmonic coupling, with the ordinary masked mean position
\(\bar x=N^{-1}\sum_j x_j\),

\[
-\frac{\kappa}{N}\sum_{j\ne i}(x_i-x_j)
= -\kappa(x_i-\bar x).
\]

This follows directly from the force equation in
[physics.py](../../ripii/world/physics.py). It is an ordinary particle-count mean,
not a mass-weighted mean. Mass enters the acceleration through \(F_i/m_i\).

Consequently, this task has a particularly strong simple explanation: a learned
mean-based or parametric control can express the long-range functional form
without learned grouping. Its unknown quantities must be fitted only on the
permitted training/development data. The report must disclose any supplied
internal law or coefficient and any shared generator constant that makes a
coefficient inferable from the known wall environment.

A hierarchy that beats only the radius-limited local model has not answered the
second question. The equally informed full-pair/global control and simple
parametric control are necessary competitors. A hierarchy loss to either is a
useful result and must remain visible.

### The reference simulator already knows the answer

The full known-law simulator is an oracle/reference. It receives the actual
generating dynamics and is useful for numerical parity, units, and sanity checks.
It is not a fitted learned competitor and is not evidence of learned
generalization. Report oracle, prior-only, and fitted-model rows separately.
The learned-model decision must refer to the predeclared competitive controls.

There is no justified need to make the simple control artificially large or
undertrain it to resemble a neural network. Record its actual parameters and
fitting method. Match relevant information and budgets, disclose functional-prior
differences, and interpret remaining capacity differences honestly.

## Assumptions that require explicit tests or qualifications

- **Physical balances:** central internal pair forces may conserve momentum and
  torque. The full forced, wall-contact, dissipative world need not conserve total
  momentum or energy. Zero action is not equivalent to zero external force.
- **Geometry:** arbitrary translation of particles with a fixed arena is not a
  symmetry of wall dynamics. Restrict claims to the internal mechanism and valid
  joint transformations or arena symmetries.
- **Variable count:** masks and padding must affect all models identically.
  Existing ordinary splits use two to four objects; the capacity-eight
  more-objects split uses five to eight. Explicit single-object cases are
  diagnostics unless the protocol separately makes them training data.
- **Mass and coarse denominators:** live masses at least 0.7 avoid the historical
  simulator's 0.1 mass floor. They do not remove every possible coarse-mass,
  inertia, anchor-tie, or empty-group issue. Universal algebraic claims require
  stronger evidence than generic-scene tests.
- **Gradients:** meaningful state/action/parameter paths, numerical checks away
  from declared nonsmooth points, and reload/resume behavior must match the
  specification. A passing forward test is insufficient.
- **Rollouts:** one-step or teacher-forced accuracy is not free-rollout accuracy.
  Freeze training horizons, noise, selection, evaluation horizons, and
  model-specific fitting procedures before protected outcomes are examined.
- **Measurement:** pooled object RMSE and mean scene RMSE estimate different
  quantities. Freeze weighting across coordinates, horizons, scenes, data seeds,
  and OOD splits, including ratio denominators near zero.

These are prospective evidence requirements. None is marked repaired merely
because the design proposes a remedy.

## Register semantics and execution gates

[FLAWS.csv](FLAWS.csv) contains forty stable identifiers, CLOSURE-F01 through
CLOSURE-F40. Each row records the flaw, cause, severity, repair, required evidence,
disposition, applicability, and source basis.

**Severity** expresses the impact on the named claim:

- **BLOCKING:** an unresolved defect would invalidate the specified comparison,
  evidence binding, or interpretation. It must be resolved before making that
  claim; this is not a new permission request.
- **MAJOR:** the issue materially limits interpretation or release scope. Some
  such issues are deliberately retained as limits of this bounded study.

**Disposition** records the present review state:

- **OPEN:** a planned control or verification still needs retained evidence.
- **ACCEPTED_LIMITATION:** the limit remains real and constrains the conclusion.
  Acceptance is not proof that the broader issue is solved.
- **HISTORICAL_UNRESOLVED:** predecessor evidence remains missing and is not
  reconstructed or replaced by successor results.

A row may be closed only by linking the exact implementation/protocol revision
and relevant retained evidence. A plan, manuscript assertion, unrelated passing
test, or checksum alone does not close it. Preserve the review history and
explain any newly discovered flaw or changed disposition.

The useful sequence of gates is:

1. **Before the development matrix:** bind the complete model and data contracts,
   factorial and control grid, seed identities, finite resource/trial caps,
   fitting procedures, metrics, stopping/failure rules, and artifact schema.
   Complete the bounded implementation and end-to-end sanity evidence needed for
   valid execution.
2. **Before the protected reveal:** freeze the exact selected configurations,
   source identities, untouched seed allocation, primary comparisons, uncertainty
   method, multiplicity treatment, and analysis. Record the reveal. Earlier
   development outcomes cannot be relabeled protected evidence.
3. **Before a scientific conclusion:** account for every planned cell, including
   nonfinite, timed-out, and incomplete cells; regenerate analysis from retained
   outcomes; report the actual independent units and precision; bind every
   numerical claim and figure/table to its source.
4. **Before broader claims:** separately obtain the external, scaling,
   convergence, independent-reproduction, and release evidence those claims
   require. Those are not automatically added to the v1 compute budget.

The exact numerical limits and statistical decisions belong in the frozen
protocol. A small protected sample can remain inconclusive despite impeccable
execution. A negative valid comparison is not an invalid experiment merely
because the model performs poorly.

## Limits that remain even after a successful diagnostic

The following boundaries are deliberately unresolved by v1:

- Official GNS/EGNN/SEGNN, MeshGraphNets, and other published baseline integrations
  are not supplied by the internal model names. The
  [pinned source registry](../baselines/official_sources_v1.json) explicitly records
  integration gaps. This study cannot establish superiority over those methods.
- No task-valid external dataset comparison is executed by using this simulator.
  The [LagrangeBench source card](../../docs/data/LAGRANGEBENCH_LDC2D.md) describes a
  2,708-particle task that cannot be silently truncated to this interface. The
  prior [NRI adapter](../../docs/data/NRI_ADAPTER.md) concerns a separate external
  synthetic simulator and is not real-world validation.
- A small fixed training budget does not establish convergence, optimal tuning,
  a universal model ranking, or the effect of every possible optimizer choice.
  The amount of statistical evidence is limited by actual independent data
  generation and scene units, not inflated by timesteps or nested model seeds.
- At most eight objects in a fixed two-dimensional world cannot establish
  large-N scaling, subquadratic runtime, 3D behavior, unseen-law identification,
  arbitrary boundary or dt generalization, noisy/missing-observation robustness,
  or real-world utility. No adaptive-computation claim is possible without a
  router, which is deliberately absent.
- Timing and memory from one machine describe the declared measurement scope.
  Similar parameter counts or equal updates are not a portable efficiency proof.
- An author-run verifier checks what it actually implements. Integrity and
  completeness verification are not independent scientific replay. Independent
  reproduction remains unestablished until separately retained evidence exists.
- The original seventy-five-run MR archive remains unrecovered. Its reported
  negative result is preserved; this study cannot quantify how much of that
  historical error was caused by the new diagnosed mismatch.
- License, authorship, and submission metadata remain separate release
  requirements, as recorded in [LICENSE_STATUS.md](../../LICENSE_STATUS.md) and
  [PAPER_STANDARD.md](../../PAPER_STANDARD.md). They do not prevent authorized
  local diagnostic work, but no submission-ready or licensing claim follows from
  finishing the numerical study.

## Terminal interpretation

The result may support the declared closure diagnosis while finding hierarchy
unnecessary. It may show mixed benefits across the prespecified conditions. It may
be negative, too imprecise to resolve the practical threshold, or invalid because
of a specific verified defect. Report those possibilities separately for the
declared questions and retain all failures.

No outcome licenses an indefinite search for positive results. Any material
change after the frozen evidence is inspected becomes a separately versioned
successor with a new hypothesis and evidence boundary. The existing
[negative-result synthesis](../NEGATIVE_RESULT_SYNTHESIS.md), frozen protocols,
and historical result artifacts remain intact.
