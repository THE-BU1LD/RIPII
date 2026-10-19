# Closure study v1: development gate audit

Date: **2026-10-19 UTC**, after development completion and before the
confirmation freeze/reveal. Reviewer: separate evidence-audit agent; read-only
inspection of retained development artifacts, verification receipts, the
[protocol](protocol.json), [FLAWS.csv](FLAWS.csv), and
[prospective review](FLAW_REVIEW.md). This audit ran no training, model replay,
or new outcome-bearing computation.

## Gate disposition

**PASS for valid-pipeline development eligibility. No concrete development
blocker was found.** The complete, finite, verified development grid is eligible
for the unchanged protected confirmation phase under the existing protocol.
Eligibility does not depend on favorable development accuracy. The reported
development H2 disposition remains **NOT_TESTED**; no confirmatory p-value is
computed or inferred here.

The next existing gate is the exact source/commit/environment/protocol freeze
before generating confirmation tensors. At this audit's start, the execution
owner reported that confirmation had never been generated. This audit did not
access protected data and does not substitute for the freeze and first-reveal
receipt. No architecture, seed, endpoint, budget, or decision-rule change is
recommended from these observed outcomes.

## Evidence identity and verification

The audited phase is
[`runs/closure_study_v1_20261019/development`](../../runs/closure_study_v1_20261019/development).

| Binding | Retained identity |
| --- | --- |
| Executed source commit | `6b99a71768e429ace84116116a7ca8ffca06f35f` |
| Source fingerprint, 50 retained files | `37e6ad8b22a2a50cfcfb3ee2fb482d5c3c02b0be455c9bd8188ab09dc6c65aa6` |
| Canonical protocol SHA256 | `41e20eb8a6342b41c4183dd33cd70ff6b5bb743c8d461c70d387f54bfa31801f` |
| Evidence manifest SHA256, 440 covered files | `83ac5d7f075d210b50ce3b98e7581c10111fe58e094fe5d465e3ba1f3c04d76a` |

These identities come from the retained `source.json`, `protocol.json`,
`status.json`, and `manifest.json`. The retained protocol equals the current
prospective protocol. The manifest digest was also checked during this review.
Source snapshots, environment/dependency records, dataset tensors and identities,
training traces, selected/final checkpoints, raw per-scene statistics,
predictions, and generated analysis remain in the phase package.

The execution owner's
[replay receipt](../../runs/closure_study_v1_20261019/verification/development_replay.json)
records verification of all 440 files, 132 clean-split raw-metric calculations,
132 exact selected-model/oracle trajectory replays, and regenerated statistical
analysis. These counts cover 33 arm results across four clean evaluation splits.

The separately implemented NumPy
[verification receipt](../../runs/closure_study_v1_20261019/verification/development_independent.json)
passes with zero errors: 165 prediction splits, including `noisy_test`, and
8,862 scalar/array comparisons. Float64 recomputation versus retained float32
arithmetic has maximum absolute discrepancy `4.409302285068861e-06`; the largest
discrepancy relative to its allowed tolerance is `0.17050426283214967`.
Its declared tolerances are absolute `2e-7` and relative `5e-6`. Its script SHA256
is `1bd31624ddbe6b7407e050c3b1aedbb0a62559b3a30c574734e395d0f4f4c9b3`.
This verifier is explicitly outside the original retained scientific source; it
recalculates saved-array measurements and the prescribed analysis without
retraining, replaying models, or changing the published H2 classification.

These are complementary author-environment verification receipts and a separate
artifact review. They do not establish external hardware reproduction,
independently regenerated training, or authenticated historical provenance.

## Grid, data, and finite execution

| Check | Observed evidence |
| --- | --- |
| Independent development units | Exactly `(71011,19011)`, `(71023,19023)`, `(71039,19039)` as data/model seed pairs |
| Planned arm grid | 11 results per pair: all ten protocol arms plus `global_compute`; 33/33 complete |
| Terminal states | Every arm follows `planned → running → complete`; root phase complete |
| Evaluation completion | Every arm retains `test`, `more_objects`, `composition`, `fast`, and `noisy_test` |
| Analysis validity | `complete=true`, three observed/expected units, empty validation errors and missing lists |
| Finite outputs | No nonfinite floating values in phase-result JSON; retained predictions/metrics covered by both verification receipts |
| Base training budget | Every fitted base arm completes 120 updates; prior/oracle complete zero fitting updates |
| Phase wall time | `130.52707034600462` seconds within the executed 700-second development cap |
| Remaining outcome budget | `1969.4729296539954` seconds of the frozen 2100-second development-plus-confirmation budget |
| Execution environment | CPU, one Torch thread, deterministic algorithms; Torch `2.14.0+cpu`, NumPy `2.5.3`, Python `3.12.14` |

Wall time includes setup, data preparation, training, evaluation, and analysis;
it excludes final manifest hashing, as the phase record states. Verification
cost is separate. Peak process RSS is 328,596 KiB (320.895 MiB), a process-level
high-water mark rather than a model allocation measurement.

All 18 retained DatasetSpecs match the declared seed, full Physics, capacity,
scene count, and horizon: per unit, 96 training scenes of horizon 16, 16
validation scenes of horizon 32, and 24 scenes of horizon 32 in each of four
evaluation splits. All 624 initial-scene hashes are distinct across these
development datasets. These hashes support identity/disjointness checks; they
do not create 624 independent experimental replicates.

The dataset freeze timestamp, `2026-10-19T14:15:07.191996Z`, precedes the first
arm's running timestamp, `2026-10-19T14:15:07.195761Z`. Each data manifest records
creation before training. No incomplete or failed cell appears in this retained
phase. This observation is scoped to the audited phase, not a blanket assertion
about every possible prior attempt.

## Controls and compute coverage

The hierarchy, equal-update global arm, and extended global arm each have
**2,386 trainable parameters**. Local factorial arms have 1,521, the analytic-mean
arm 1,522, the law control one, and prior/oracle zero. The simpler control is not
artificially enlarged to match the neural counts.

| Data/model seed pair | Hierarchy optimization seconds | Extended global seconds | Coverage ratio | Global completed updates |
| --- | ---: | ---: | ---: | ---: |
| 71011 / 19011 | 9.641291 | 9.657804 | 1.001713 | 219 |
| 71023 / 19023 | 8.646518 | 8.648058 | 1.000178 | 201 |
| 71039 / 19039 | 9.683944 | 9.720203 | 1.003744 | 227 |

Each extended global arm covers the hierarchy's measured optimization time and
stops below the 480-update ceiling. This metric includes window sampling,
forward/backward execution, clipping, and optimizer steps; it excludes
validation/evaluation. Equal-update and time-covered results are both retained.
The extension is a continuation of its base global fit, not another independent
seed. Validation-selected checkpoints may precede the final completed update.

The corrected competitive arms share known wall/drag closure and four substeps.
The law control additionally receives the exact contact functional form and
fits only the coupling scalar; its stronger functional prior is intentional and
must remain disclosed. Its retained fitted couplings are approximately
0.996033, 0.994925, and 0.994516. The full known-law simulator is a separate
zero-error oracle, outside learned-model log-ratio comparisons.

## Descriptive development evidence

Values below come from the frozen
[development analysis](../../runs/closure_study_v1_20261019/development/analysis/results.md).
Ratios are geometric means of three paired hierarchy/control OOD ratios;
within each pair, OOD is the equally weighted mean of the three regime RMSEs.

| Control | Hierarchy/control OOD ratio | Descriptive bootstrap interval |
| --- | ---: | --- |
| `global_compute` | 1.044326 | [1.026774, 1.057095] |
| `mean_env_s4` | 0.837954 | [0.805438, 0.861510] |
| `law_env_s4` | 104.444402 | [85.662595, 140.975980] |

The hierarchy's IID ratio versus `global_compute` is 1.192515. Development
therefore favors the time-covered global control and especially the fitted law
control, while favoring hierarchy over the analytic-mean neural control.
Local closure-on OOD means are about 0.188, versus about 0.382 with closure off;
four substeps do not improve those learned aggregate point means over one
substep in this small development sample. These are descriptive observations,
not a confirmatory H2 classification or a substitute for H1's separate
mathematical/parity evidence. With only three independent pairs, the bootstrap
intervals cannot support broad precision claims.

## Prospective flaws and remaining limits

The forty-row prospective register remains unchanged: 30 `OPEN`, nine
`ACCEPTED_LIMITATION`, and one `HISTORICAL_UNRESOLVED`. This gate audit supplies
phase-specific evidence for grid/control/failure accounting, finite budgets,
data identities, raw retention, executed-source binding, and numerical analysis
verification (notably F05–F09, F17–F19, F21–F26, and F28–F34). It does not
automatically close those rows or the separate mathematical/gradient evidence
requirements. Protected evaluation and final claim dispositions remain pending.

The accepted limits remain material: small fixed training budget without
convergence claims; fixed 2D synthetic world, at most eight objects and the
declared mass/environment contract; only the named synthetic shifts; no official
baseline or task-valid external dataset integration; no large-N efficiency,
architecture novelty, SOTA, or general law-identification claim. The single
initial-noise diagnostic does not establish broad observation robustness.
Independent external reproduction and release metadata/licensing remain
unresolved. The original 75-run MR archive remains unrecovered, and successor
results cannot attribute its aggregate errors or replace its reported negative
outcome. None of these scoped limitations creates a new efficacy gate or permits
hindsight retuning within v1.
