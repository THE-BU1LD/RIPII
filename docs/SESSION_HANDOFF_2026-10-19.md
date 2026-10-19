# RIPII engineering handoff — 19 October 2026

## Outcome and source identity

The action-conditioned inference contract is repaired and covered by regression
tests. A separate deterministic mechanics diagnostic exposes representability
obstructions from simulator drag, wall forces, and substepped force integration.
The final full suite passed 273 tests with 81.42% combined statement/branch coverage.
No new scientific matrix, trained successor, advancement decision, or readiness
assignment was made.

- Repository: `https://github.com/THE-BU1LD/RIPII`.
- Development branch: `fix/ripii-contract-audit-20261019`.
- Inspected base: `33b799fb3bb0d5215e954c5f80d2225c51a5caaa`, initially clean.
- Writer: ChatGPT root. Delegated mechanics/evidence reviewers were read-only in
  this repository. No other repository was modified.
- The commit containing this handoff is the delivery revision. Execution happened
  before that commit, against the repaired working tree on the recorded base.
  The run's source hashes and snapshots identify the executed code; its recorded
  `HEAD` must not be mistaken for an execution of the unmodified base.
- Publication is one normal descendant commit on the development branch, with
  `[skip ci]` to avoid push/PR workflow fan-out. No merge, release, deployment,
  workflow edit, scheduled task, or scientific remote job is included.

The required status/audit/protocol/manuscript documents were inspected. The named
historical `runs/ripii_mr_development_v1/report.md`, summary, manifest, and checkpoints
are absent from the checkout. The verification attempt fails on the missing
manifest. The historical 75-run, 0/60-pass, `no_advance` figures remain source-reported.
They were not reconstructed from manuscript prose. A bounded search of accessible
RIPII files is recorded separately in the evidence archive.

## What changed and why

`WorldModel.validate_inputs` now checks the complete request before executing learned
operations. It validates shapes, Boolean masks, floating state/action types, matching
dtypes/devices, nonempty batches/scenes, finite values, positive live radius/mass, and
zero state/action padding. `rollout` checks every action timestep up front, including
the otherwise empty `T=0` path. A bad later action can no longer partially execute a
rollout or update mechanism counters before rejection.

`WorldPredictor` inspects original dtypes before conversion. Numeric masks such as
`2`, `0.5`, and `NaN` cannot silently become live objects. Integer states/actions are
rejected; valid floating wrapper inputs still convert to float32. The raw model
continues to support matching double tensors when its parameters are double.

The learned dynamics equations, parameters, state-dict keys, checkpoint schema, and
historical model namespaces are unchanged. The stricter contract intentionally
rejects previously accepted malformed requests. It adds validation work, including
full-sequence scans and possible device synchronization; no unchanged-latency claim
is made.

The new tests contain 52 contract cases and 10 mechanics cases. Original and repaired
source both replay the retained four-scene float32 fixture for all eight variants:
the eight predictions and all 208 parameter-gradient tensors per revision match the
same baseline tensors bitwise. The replay script loads only restricted tensor/state-
dict payloads and checks that the original class's project dependencies are unchanged.
This is a bounded compatibility check, not a universal numerical or runtime proof.

## Mechanics diagnosis

Write scene momentum as `P = sum(m_i * v_i)` and mass-weighted position as
`Q = sum(m_i * x_i)`. Under their intended conservative constraints, both MR variants
apply:

```text
P_next = P + dt * sum(action_i)
Q_next = Q + dt * P_next
```

The simulator includes drag and wall impulses in addition to the supplied action.
A single object has no internal pair impulse through which any learned weights can
represent those missing external effects. Even with drag and contact disabled,
constant force exposes an integration difference: for four semi-implicit substeps,
the first position increment has acceleration coefficient `(4 + 1)/(2 * 4) = 0.625`;
MR's single semi-implicit update has coefficient `1`. Final velocity agrees in that
control. Setting the simulator to one substep removes that position obstruction.

For a discrepancy `D` in either mass-weighted sum, Cauchy–Schwarz gives minimum
squared coordinate error `||D||^2 / sum(m_i^2)`. The aggregate RMSE divides the sum
of these minima by `2 * horizon * total_live_objects` before taking the square root.
These are lower bounds under the stated constraints, not fitted-model scores.

The [generated report](../research/results/development/ripii_contract_audit_20261019/report.md)
contains every measured bound and routing probe. The nine fixed fixtures cover free
motion, two masses under drag, wall contact, force with four versus one substep,
internal contact, long-range coupling, and five objects with shifted masses. Both
variants run for eight steps with disclosed untrained weight interventions and no
optimizer updates. The largest checked momentum and position constraint residuals
are respectively `1.1102230246251565e-16` and `1.6653345369377348e-16`.

Free-motion/drag/constant-force analytic references cover all eight observations.
The wall reference checks only the first observation, while that fixture remains in
its stated linear contact regime. Internal contact and symmetric coupling have zero
net-force bounds. A zero bound does not mean the individual trajectories are accurate.
Multi-object conclusions remain conditional: denominator floors and degenerate/tied
coarse assignments prevent an unconditional all-scene conservation guarantee.

The existing `ripii_mr_v2` was already present at the inspected base. It was not
implemented by this change. Coarsener hooks observe v1 off/on execution counts `1/1`
and v2 counts `0/1`. At nonsaturated route logits `-1/+1`, the v2 off scene has zero
router gradient from the actual simulator state-prediction loss; the on scene has a
nonzero gradient. This localizes a gradient-reachability limitation. It does not show
training collapse, measured latency savings, or improved predictive quality.

No calculation estimates how much of the historical MR error comes from these
obstructions. The missing original trajectories and checkpoints prevent that claim.

## Actual environment and budget

Execution used Linux x86_64, Python 3.12.14, PyTorch 2.14.0+cpu, nine logical CPUs,
and approximately 9.73 GiB physical RAM. The retained resource observation had more
than 18 GiB free disk and 8 GiB available RAM. CUDA and MPS were unavailable. This was
not execution on Ryan's Apple Silicon machine or on Kaggle.

There was one sequential verification/compute lane with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. Full tests had a 600-second limit; the diagnostic had a
120-second limit and ran in 1.372887 seconds internally (3.224471 seconds including
process startup). Its device is CPU, dtype float64, hidden width 16, seed 1907, maximum
object count eight, horizon eight, and scientific training-update count zero. Existing
smoke/resume tests performed their own bounded test-only optimizer fixtures.

The venv inherits system site packages. Core runtime versions were completed to the
Python 3.12 lock selections: NumPy 2.5.3, Matplotlib 3.11.1, PyYAML 6.0.3, tqdm 4.70.0,
and the CPU build of Torch 2.14.0. Tooling included pytest 9.1.1, pytest-cov 7.1.0,
Ruff 0.16.6, and mypy 1.20.2. This is not an exact `uv sync --locked` environment;
the complete installed-distribution freeze is retained.

The first full-suite attempt had four collection errors because tqdm was missing.
After completing dependencies, all 273 tests passed with zero failures, errors, or
skips. The default mypy invocation targets Python 3.10 and fails on Python 3.12-only
syntax in the installed NumPy stub. `mypy --python-version 3.12 ripii` passes all
43 package files. Python 3.10 type-checking and the hosted CI environment remain
unverified in this session. The earlier failures are preserved in the archive.

An available browser integration timed out during initialization, so no authenticated
Kaggle account, notebook, accelerator, remaining quota, dataset access, or current
job state was established. GitHub Actions workflow triggers and recent public run
states were inspected; billing allowance was not exposed by the available connector.
No allowance-exhaustion claim is made. The commit skip instruction follows
[GitHub's documented push/PR behavior](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs).
It does not waive future merge requirements or change existing scheduled workflows.

## Commands and results

All commands below run from the repository root. The archive's
`verification-commands.json` contains the exact final argument vectors, environment
overrides, exit codes, and elapsed times. Earlier iteration logs and JUnit files are
also retained; `test-counts.json` extracts their actual counts.

| Check | Command / selection | Result |
|---|---|---|
| Existing targeted baseline | `python -m pytest tests/test_world.py tests/test_multiresolution.py tests/test_inference_api.py` | 83 passed on the original source |
| New contract cases before repair | `python -m pytest tests/test_world_contract.py` | 42 failed, 10 passed; expected regression evidence |
| Final complete suite | `.venv/bin/python -m pytest -p pytest_cov --cov=ripii --cov-report=term-missing --junitxml=runs/engineering_contract_audit_20261019/full-tests.xml` | 273 passed; 81.42% coverage; exit 0 |
| Original/repaired valid-input replay | `.venv/bin/python runs/engineering_contract_audit_20261019/replay_compatibility.py runs/engineering_contract_audit_20261019/baseline-valid-tensors.pt` | eight variants, 208 gradient tensors per revision match; exit 0 |
| Lint | `.venv/bin/python -m ruff check .` | pass; exit 0 |
| Compile | `.venv/bin/python -m compileall -q ripii scripts tests` | pass; exit 0 |
| Configured type target | `.venv/bin/python -m mypy ripii` | installed NumPy stub/Python 3.10 target mismatch; exit 2 |
| Actual runtime type target | `.venv/bin/python -m mypy --python-version 3.12 ripii` | 43 files pass; exit 0 |
| Existing CPU performance guard | `.venv/bin/python scripts/check_ci_performance.py` | 4,180 parameters; 0.620506 ms mean forward on this host; exit 0 |
| Wheel build | `.venv/bin/python -m pip wheel --no-build-isolation --no-deps --no-index --wheel-dir runs/engineering_contract_audit_20261019/wheel .` | wheel built; exit 0 |
| Portable historical evidence | `bash scripts/verify_artifact.sh` | pass; omitted checkpoints explicitly excluded; exit 0 |
| Original MR matrix | `.venv/bin/python scripts/run_ripii_mr_matrix.py --output runs/ripii_mr_development_v1 --verify-only` | missing original manifest; exit 1 |
| Fresh mechanics diagnostic | `.venv/bin/python -m ripii.world.mechanics_audit --output runs/engineering_contract_audit_20261019/mechanics_v1` | nine fixtures complete; exit 0 |
| Diagnostic integrity/completion | `.venv/bin/python -m ripii.world.mechanics_audit --verify runs/engineering_contract_audit_20261019/mechanics_v1` | 52 artifacts verified; exit 0 |

The full test suite preceded two wording-only edits to the diagnostic's docstring
and generated report. The exact tested/final source hashes and that diff are retained
in `source-versions.json` and `post-suite-documentation-only.diff`. The final standalone
diagnostic, source snapshots, type check, lint, and wheel include the clarified text.
The performance guard is a generous regression alarm, not an efficiency study.

## Retained package and source-to-evidence map

The compact package is
[`research/results/development/ripii_contract_audit_20261019`](../research/results/development/ripii_contract_audit_20261019).
It contains a generated report, machine-readable summary, archive, and manifest. The
archive retains the complete final diagnostic, initialized weight files, package source
snapshots, protocol, literal fixture identities, targets, predictions, measurements,
status transitions, logs, environment freezes, wheel, compatibility tensors/replay,
regression source snapshots, and the six-file implementation patch. It contains only
this engineering session's synthetic fixtures and evidence, with no private datasets
or historical large checkpoints.

| Source | Execution/evidence | Supported statement |
|---|---|---|
| `ripii/world/models.py`, `inference.py` | `test_world_contract.py`, retained before/after results, original-source replay | contract rejection and bounded valid-input compatibility |
| `ripii/world/mechanics_audit.py` | `protocol.json`, `measurements.json`, `report.md`, two untrained state dicts | explicit synthetic representability bounds and execution/gradient observations |
| Unchanged `multiresolution.py`, `physics.py`, `experiment.py` | source snapshot hashes, analytic references, coarsener hooks, actual state-prediction loss | mechanism interpretation tied to the inspected implementation |
| `test_mechanics_audit.py` | full suite, source snapshots | finite multistep state/action/parameter gradients, bounds, process-state restoration, and tamper rejection |
| Existing historical capsule verifiers | `portable-evidence.log` | portable integrity only; no omitted checkpoint replay |

The diagnostic manifest SHA-256 is
`d8f16d4a587adb13a35ce201b5bf272bf43dbe5aedfd65c697b3a202ecfc1d6a`.
The outer manifest records the archive digest. Hash verification checks integrity and
completion; it is not independent reproduction or an authenticated author signature.

## Reproduction, including a Kaggle CPU handoff

Check out the immutable delivery commit containing this handoff. Install the project
with a suitable Python environment (`uv sync --locked --extra dev` is the repository's
standard path). To match this run's CPU variant on Python 3.12, use the recorded
package versions and full freeze; record any platform/dependency deviations.

To verify the delivered archive, extract it to a fresh directory, keeping its
`engineering_contract_audit_20261019` root, then run:

```bash
.venv/bin/python -m ripii.world.mechanics_audit --verify \
  /absolute/fresh/extraction/engineering_contract_audit_20261019/mechanics_v1
.venv/bin/python /absolute/fresh/extraction/engineering_contract_audit_20261019/replay_compatibility.py \
  /absolute/fresh/extraction/engineering_contract_audit_20261019/baseline-valid-tensors.pt
```

The executable `python -m ripii.world.mechanics_audit` is the ready-to-run diagnostic
for a connected Kaggle notebook as well. Before launching there, inspect and record
the authenticated account/notebook URL, existing jobs, accelerator setting, quota,
internet, and data access. Select **CPU / no accelerator** for this diagnostic. Clone
this repository and pin the delivery commit, use a private notebook, and install the
environment in that checkout. The nine fixtures require no external dataset.

From the pinned checkout, these shell commands produce a fresh, retrievable bundle:

```bash
set -euo pipefail
audit_root=/kaggle/working/ripii_mechanics_replay_001
mkdir "$audit_root"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
git rev-parse HEAD > "$audit_root/source_commit.txt"
.venv/bin/python -m pip freeze > "$audit_root/environment.txt"
timeout 120 .venv/bin/python -m ripii.world.mechanics_audit \
  --output "$audit_root/run" > "$audit_root/launch.log" 2>&1
.venv/bin/python -m ripii.world.mechanics_audit \
  --verify "$audit_root/run" > "$audit_root/verification.json"
tar -czf "$audit_root.tar.gz" -C /kaggle/working ripii_mechanics_replay_001
sha256sum "$audit_root.tar.gz"
```

Use a new directory name for each repeat; the runner refuses existing outputs. If a
run fails, retain the partial directory and its logs/status before making another
attempt. Download the `.tar.gz` from the notebook's output/files panel, retain its
printed hash and actual notebook/version identity, and rerun integrity verification
after extraction. These are launch/retrieval instructions; no Kaggle execution or
unattended monitoring has occurred in this session.

## Exact changed files

1. `README.md`
2. `EVIDENCE_LEDGER.md`
3. `docs/USAGE.md`
4. `docs/SESSION_HANDOFF_2026-10-19.md`
5. `ripii/world/models.py`
6. `ripii/world/inference.py`
7. `ripii/world/mechanics_audit.py`
8. `tests/test_world_contract.py`
9. `tests/test_mechanics_audit.py`
10. `research/results/development/ripii_contract_audit_20261019/report.md`
11. `research/results/development/ripii_contract_audit_20261019/summary.json`
12. `research/results/development/ripii_contract_audit_20261019/evidence.tar.gz`
13. `research/results/development/ripii_contract_audit_20261019/manifest.json`

## Go/no-go and next action

**GO for engineering review of the contract repair and retained diagnostic. NO-GO
for another frozen-architecture matrix, a superiority claim, or research advancement
on this evidence.** The mechanism obstruction merits diagnosis, not a claim that a
new architecture has won. No historical thresholds, checkpoints, source meanings,
or negative decisions were changed.

Status distinctions: **IMPLEMENTED** and **LOCALLY_TESTED** apply. **REMOTELY_EXECUTED**
and **INDEPENDENTLY_REPRODUCED** do not apply to the experiments. Scientifically
supported statements are limited to the stated deterministic construction properties
and synthetic fixture observations; learned predictive advantage remains unestablished.

Remaining gaps are the original MR artifact archive; CUDA/MPS/Kaggle execution;
the Python 3.10/hosted type-check path; and broader mass-floor/tied-coarsening behavior.
The simulator's live-mass floor and degenerate coarse-group semantics were identified
as separate concerns and preserved because changing them would change dynamics or
historical interpretation. The original MR matrix verifier also merits a separate,
version-aware completeness audit after the original archive is recovered; its runner
source is itself part of newer provenance contracts.

**Next single action:** recover the original 75-run MR archive with its recorded
source and verify it in a fresh evidence directory. Only those retained trajectories
can support an aggregate attribution of the historical error gap to the diagnosed
constraints.
