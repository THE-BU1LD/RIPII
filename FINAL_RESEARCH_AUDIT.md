# Final Research Audit

Paper: *Learned hierarchy does not improve a controlled synthetic object-dynamics benchmark*  
Repository: RIPII  
Commit: `ec69f39f997f8922963d0850a36ef84c91931174`  
Target Venue: Unspecified serious ML/scientific-ML conference or journal  
Date: 2026-09-13  
Audit basis: actual code, frozen protocols, retained manifests/capsules, manuscript,
tests, data records, prior audit, and the completed post-correction study.

## Executive Decision

**Score: 71/100 — major scientific revision required.**  
**Readiness level: P2 — repaired-method evaluation complete; submission evidence incomplete.**  
**Recommendation: do not submit. Perform the external, powered, compute-matched program
and official baseline comparisons before reconsidering submission.**

No fabrication, citation invention, obvious train/test leakage, or hidden manual primary
result editing was found. That is necessary but insufficient. Submission remains blocked
by omitted required strong-baseline results,
inadequate external validity and statistical power, no independent reproduction, and
unresolved release ownership/disclosure metadata.

## Scientific Question

Does learned soft coarse-to-fine grouping improve object-state dynamics prediction and
OOD rollout over simpler local graph and global-context controls under matched data,
initialization seeds and approximately matched model capacity?

The technical obstacle is whether learned aggregation discovers useful physical scale
structure rather than adding a flexible, weakly identifiable route that harms
optimization or duplicates ordinary global context.

## Main Hypothesis

The tested hierarchy should reduce held-out rollout error relative to a flat interaction
graph and a direct global-pooling control, and the improvement should persist across
paired seeds and controlled OOD regimes without relying on greater capacity.

For repaired RIPII 0.2, the frozen development hypothesis was narrower: the complete
`base` objective had to lower reconstruction MSE by at least 5% on every paired seed
versus both reconstruction+KL and the capacity-matched plain autoencoder. The completed
protocol returned its predefined failure decision, `no_advance`.

## Main Contribution

The defensible contribution is a well-instrumented negative-result research system:

1. a from-scratch implementation of learned soft hierarchy and multiple simpler controls;
2. frozen, manifest-bound experiments that repeatedly fail to show hierarchy-specific
   benefit in the tested synthetic settings;
3. mechanistic diagnostics showing active grouping but poor task utility and conflicting,
   unequal legacy objective gradients;
4. reproducibility engineering that preserves failures and prevents unsupported claims.

No new theorem, fundamentally new algorithm, state-of-the-art result, or validated
renormalization interpretation is established.

## Strongest Evidence

The five-seed world-v3 development study is the strongest completed evidence. Graph has
lower mean ID position RMSE (0.0904 ± 0.0075) than multiscale (0.0983 ± 0.0097) and wins
the aggregate OOD comparison on all five paired seeds. The content-verified capsule binds
the raw summary, protocol digest and `no_advance` decision.

This supports a bounded negative statement about this implementation and simulator. It
does not establish that learned hierarchy is generally ineffective.

## Strongest Baseline

The strongest executed task-level control is the flat continuous graph model. It shares
the object-state contract, training data, split, validation procedure and paired model
seeds while removing learned coarse grouping. Global pooling is the decisive control for
whether hierarchy adds more than generic scene-wide context.

An E(2)-equivariant continuous control is implemented and engineering-tested but has no
publication-scale result. Official GNS, EGNN/SEGNN, constraint-based GNS, MeshGraphNet/
BSMS and energy/Lagrangian baselines are not yet fairly integrated and executed. This is
a submission blocker.

## Main Quantitative Result

| Model | ID position RMSE, mean ± sample SD | OOD interpretation |
|---|---:|---|
| Flat graph | **0.0904 ± 0.0075** | Lowest mean error in each reported OOD regime |
| Global pool | 0.0920 ± 0.0070 | Simpler global-context control |
| Multiscale | 0.0983 ± 0.0097 | Loses graph comparison on 5/5 paired seeds |

Multiscale's mean OOD relative improvement versus graph is -13.05%; versus global pool
its mean OOD advantage is only 1.58%, with mixed paired direction and worse mean ID
error. These are descriptive five-seed development statistics, not population inference.

## Most Important Ablation

The fixed 39-run objective study compares reconstruction+KL with eleven individual
auxiliary additions and the complete legacy objective. Every addition worsens the
fixed-budget reconstruction metric on every tested seed; the complete objective is worse
by 26.4%, 60.0% and 89.5% across the three paired seeds.

The original interpretation was limited because 30 updates and three seeds characterize
early optimization, not convergence. The completed 180-cell study expanded this to five
seeds and 600 updates across component removals, objective additions, hierarchy depth,
projector count, graph sparsity, VQ controls and a capacity-matched plain autoencoder.
It again returned `no_advance`: base mean reconstruction MSE was 0.08410, compared with
0.04687 for reconstruction+KL and 0.05573 for the plain autoencoder; both controls beat
base on every paired seed. The best observed secondary mean, 0.04329 for
reconstruction+KL plus equivariance, is exploratory rather than a new positive claim.

## Strongest Robustness Result

The negative hierarchy result persists across more objects, held-out radius/mass
composition and faster motion in the bundled simulator. Post-result stratification also
finds the deficit in contact, near-contact, forced and free-flight subsets. An independent
NRI Springs/Charged development extension does not advance hierarchy, and a prospective
long-range coupling intervention favors simple global context rather than hierarchy.

These are useful robustness signals across synthetic generators and interventions, but
not empirical-domain or adequately powered external validation.

## Most Important Failure

The proposed hierarchy fails its own advancement rule. Its soft assignments remain
active, so the failure cannot be dismissed as a completely dead grouping module; active
assignments simply do not yield superior predictive structure. The simplest current
explanation is that local graph interactions already encode the useful prior while soft
coarse aggregation introduces optimization burden or unhelpful mixing.

The post-correction runner also recorded an operational broken-pipe failure after 20
valid cells. No cell evidence was lost; execution resumed from verified manifests in a
persistent tmux session and completed all 180 cells. Operational recovery must not be
confused with scientific success: the completed study failed its scientific gate.

## Primary Limitation

Evidence is dominated by low-dimensional synthetic dynamics with known object states.
There is no learned perception, noisy empirical trajectory benchmark, high-dimensional
observation study, or demonstrated large-particle scaling. The authenticated
LagrangeBench 2D LDC data is present, but its 2,708-particle task cannot be silently
truncated into RIPII's small rigid-disc contract; a scientifically valid adapter and
scalable method are still missing.

## Mathematical Risk

The implemented equations are broadly coherent, and masking/statistical aggregation are
audited. The principal mathematical risks are interpretive:

- “projective renormalization” lacks an RG flow, fixed point, universality or semigroup
  result;
- soft grouping is not demonstrated scale separation;
- most learned world variants do not guarantee E(2) equivariance or conservation;
- fixed residual scales 0.5 and 0.05 rely on implicit normalization and allow a direct
  position bypass;
- the legacy uncertainty-weighted multi-loss objective is weakly identifiable and has
  empirically conflicting gradients;
- graph neighbor discovery remains quadratic even after sparse message gathering.

There is no central theorem whose proof explains the empirical result. The manuscript
must not imply otherwise.

## Experimental Risk

- The repaired-objective study is complete but remains a five-seed, single-generator
  synthetic development experiment.
- Five paired seeds cannot attain conventional two-sided exact sign-flip significance;
  current inference is descriptive.
- Equal updates and similar parameter counts do not establish equal compute or equal
  tuning opportunity.
- Some comparisons share one generated dataset seed, so model-seed variation does not
  capture dataset-generation uncertainty.
- Required recent/official task baselines are not yet executed under a shared budget.
- Hyperparameter sensitivity, data-efficiency curves, larger-N scaling and device
  tolerance are incomplete.
- Confirmatory external test data has not been run under a prospectively powered design.

## Reproducibility Risk

Positive engineering evidence includes locked dependencies, immutable protocols,
source hashes, deterministic seeds, validation-only selection, raw JSON/CSV, checkpoints
in full runs, exact manifests, resumable cells, content-verified capsules, generated paper
tables, branch coverage and archive tooling.

Remaining risks are decisive for P5:

- compact capsules omit checkpoints;
- the completed full run is preserved in a 4,803,673,602-byte local archive with a
  2,183-file internal manifest and SHA-256 sidecar, but has no independent mirror;
- no independent clean-machine reproduction exists;
- CUDA is uncertified and the first scheduled MPS evidence is pending;
- no tagged immutable release or DOI exists;
- no owner-approved license grants reuse rights;
- the historical runner PID probe can report false under sandbox signal restrictions,
  requiring manifest/process corroboration.

## Reviewer Attack Surface

1. Isn't this established multiscale message passing under new terminology? **Yes, the
   novelty audit finds substantial overlap; no algorithmic novelty is claimed.**
2. Why omit official GNS/EGNN/SEGNN/BSMS/constraint baselines? **They are not yet
   integrated; submission must wait.**
3. Could the graph win because it received a better compute/tuning budget? **This is not
   eliminated; equal-budget studies are required.**
4. Does the hierarchy fail only in one generator? **NRI and coupling extensions reduce
   that concern but remain synthetic, small and development-only.**
5. Could results be seed noise? **Unanimous paired direction is notable, but five seeds
   do not provide confirmatory inference.**
6. Does active grouping prove the mechanism works? **No. Activity proves reachability,
   not causal utility.**
7. Does the mathematics justify “renormalization”? **No. That interpretation remains
   unsupported.**
8. Can another lab reproduce the result legally and exactly? **Not yet: licensing,
   release packaging and independent replay are incomplete.**
9. Was the test set used for tuning? **The audited world pipeline selects checkpoints on
   validation only; no direct test-selection path was found.**
10. Could a simpler model match or beat the method? **Yes. Current evidence favors the
    flat graph, global context, simplified objective and plain autoencoder controls.**

## Hypothesis Ledger

| Hypothesis | Status | Evidence boundary |
|---|---|---|
| Learned hierarchy improves over flat graph | **REFUTED** | World-v3, five paired seeds, one bundled simulator |
| Hierarchy adds value beyond ordinary global context | **REFUTED** | Coupling and global-pool comparisons, development-only |
| Benefit persists under controlled OOD shifts | **REFUTED** | More objects, composition and fast-motion suites |
| Legacy full objective helps reconstruction | **REFUTED** | Three-seed, 30-update objective study |
| Simplified reconstruction+KL is preferable at longer budget | **SUPPORTED in development** | 180-cell study: lower MSE on 5/5 paired seeds; five-seed synthetic evidence only |
| Mechanism repairs establish scientific utility | **UNRESOLVED** | Engineering tests prove semantics, not usefulness |
| Results generalize to community/empirical benchmarks | **UNRESOLVED** | No valid LagrangeBench result or empirical dataset |
| RIPII is a novel algorithm or RG method | **UNRESOLVED / unsupported** | Closest-work audit finds major overlap; no theorem |

## Placeholder and Shortcut Classification

| Occurrence | Classification | Submission impact |
|---|---|---|
| `{{WORLD_V3_RESULTS}}` in `paper/MANUSCRIPT.md` | Build-time evidence slot, **complete/real** | Safe: exactly one slot is required, capsule verification and tests fail closed, and the PDF contains rendered values |
| `runs/*_placeholder` output paths in frozen configs | **Hardcoded operational label** | Non-scientific; output location only, not data or a result |
| NRI inserted radius/mass/action fields | **Hardcoded semantic padding, partial** | Blocks physical/property/action claims on NRI; provenance is now explicit |
| Fixed simulator coefficients and residual scales | **Hardcoded scientific choices** | Require sensitivity/normalization analysis before broad claims |
| Confirmatory external protocol | **Scaffold/unexecuted** | Explicit submission blocker; not cited as evidence |
| Baseline-expansion protocol | **Draft/scaffold** | Must be frozen before tuning or result inspection |
| Canonical model/training paths | **Complete/real and tested** | No pass-only body, `NotImplementedError`, fake metric generator or executable pseudocode found |

## Gate Matrix

| Gate | Status | Evidence / missing action |
|---|---|---|
| Important problem and precise formulation | PASS | Mathematical specification and falsifiable hierarchy question |
| Explicit assumptions and hypotheses | PARTIAL | Core assumptions exist; numbered assumption register is incomplete |
| Evaluation frozen before inspection | PASS for v0.2 | `post_correction_v02.md` is hash-bound; all 180 cells verified |
| Primary experiments complete | FAIL | v0.2 is complete; confirmatory external study remains absent |
| Required ablations complete | FAIL | Broad v0.2 grid is complete; sensitivity/data-scale studies remain absent |
| Required strong baselines complete | FAIL | Official methods pinned but not fairly executed |
| Mathematical audit | PARTIAL PASS | Equations coherent; interpretation/theory gaps remain |
| Data/leakage audit | PARTIAL PASS | No obvious leakage; external/empirical validity missing |
| Robustness and failure analysis | PARTIAL PASS | Strong synthetic analysis; no empirical/large-N robustness |
| Statistical analysis | PARTIAL | Correct paired tools; insufficient confirmatory power |
| Claim-to-artifact traceability | PASS for world-v3 | Verified capsule and generated manuscript result table |
| Reproducibility audit | FAIL | No licensed clean release or independent replay |
| Citation/novelty audit | PARTIAL | Strong targeted audit, not a current systematic review |
| Visual/final-PDF audit | PASS for current draft | Three pages rendered and inspected; content remains incomplete |
| License/authorship/disclosure gate | FAIL | `LICENSE`, `CITATION.cff`, owner metadata missing |

## Score

| Dimension | Points |
|---|---:|
| Problem importance and clarity | 8/10 |
| Formulation and mathematics | 7/10 |
| Novelty and closest-work position | 4/10 |
| Implementation integrity | 9/10 |
| Data quality and external validity | 4/10 |
| Experimental design and execution | 9/15 |
| Baseline strength and fairness | 3/10 |
| Ablations, robustness and failure analysis | 7/10 |
| Statistics and uncertainty | 4/10 |
| Reproducibility and provenance | 8/10 |
| Manuscript and claim discipline | 8/10 |
| **Total** | **71/100** |

The score cannot yield a pass because blocking scientific issues remain.

## Readiness Level

**P2 — repaired-method evaluation complete.** The project has mature negative
development evidence and strong engineering, but the full submission experiment set is
neither frozen nor complete. P3 requires every
required experiment; P4 requires the completed evidence to survive all audits; P5
requires independent reproduction; P6 requires all scientific and release gates.

## Unresolved Issues

### P0 — submission-blocking

- Freeze, integrate and execute official strong baselines under identical data,
  preprocessing, selection, tuning and compute budgets.
- Run a prospectively powered external-development and untouched confirmatory study;
  retain negative outcomes.
- Resolve software/data license, authorship, affiliation, contribution, funding and
  conflict metadata through authorized humans.
- Produce an immutable tagged release and independent clean-machine reproduction.

### P1 — required for credible scientific interpretation

- Implement a task-valid scalable LagrangeBench adapter or explicitly abandon that
  benchmark; do not truncate 2,708 particles into the current object cap.
- Add a licensed noisy empirical trajectory dataset or narrow all claims permanently to
  synthetic simulation.
- Compute-match training and report parameters, recognized FLOPs, wall time, convergence,
  inference latency, memory and hardware.
- Add dataset-seed replication, sensitivity curves, data-efficiency, larger-N scaling,
  noise strength and OOD degradation analysis.
- Convert assumptions into a numbered register with failure conditions and map every
  claim/hypothesis to an experiment/result artifact.

### P2 — major quality work

- Generate every manuscript quantitative statement, table and figure from verified
  artifacts rather than only the primary world-v3 table.
- Expand related work into technical comparisons with the closest 3–10 methods and rerun
  the search immediately before submission.
- Add independent metric cross-checks and analytic/symbolic limiting-case tests.
- Complete CUDA/MPS tolerance evidence and document energy/compute consumption.
- Expand the paper from a compact report into a question-organized methods, experiment,
  failure, discussion and appendix structure after evidence is complete.

## Recommendation

Preserve the negative result and resist post-hoc rescue. The completed v0.2 study returned
`no_advance`, so the strongest publishable direction is a carefully bounded negative
study explaining why learned hierarchy and the legacy objective fail, not a superiority
paper. Submission should be reconsidered only after the required baselines, external
powered evidence, reproducible release and human release metadata pass the gate matrix.
