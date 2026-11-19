# Prospective radial executor repair v3

Status: SOURCE_CORRECTED; VALIDATION_AND_REBUILD_PENDING_RESOURCE_ADMISSION.
This is a separately identified engineering repair of the frozen
RIPII-RADIAL-ADAPTATION-v2 executor. It is not a successor scientific hypothesis
and does not reopen a terminal scientific study. Both failed hosted engineering
attempts, the 140-fit matrix, model source, splits and decision rules remain intact.

Confirmed defects: default-stream warmup retained autograd nodes into capture;
nonfinite checking synchronized inside capture; optimized evaluation omitted gate
diagnostics; the fixed 30,000-second reservation no longer fit the cumulative cap;
checkpoints lacked an execution path for restoring optimizer/sampler/scheduler.

Repairs: one explicit training stream; detached returned diagnostics; finite
checks after every replay; old graph release at curriculum transitions; frozen
reference evaluation for gate statistics and intervention; atomic latest checkpoint
including selected model and all recovery state; fresh-process continuation;
source-bound qualification and conservative measured cost admission. Graphs are
recaptured after resume. Numerical equivalence tolerance remains atol=2e-5,
rtol=2e-4; serialized replay tests also check exact CPU generator state.

No model parameters, objective, dataset, seed, threshold or scientific decision
was changed. CPU/GPU timings and floating-point equivalence require measurement.
The new 1.25 cost safety multiplier and zero concurrency credit are conservative
engineering admission rules, not scientific changes or a performance claim.

The notebook defaults to PREFLIGHT. DEVELOPMENT_FULL and RESUME require a returned
exact-source qualification archive and a passing fresh cost profile. CONFIRMATORY_FROZEN
is refused because the project has no independent frozen confirmation protocol.
A saved checkpoint is not a successful resume until the new-process test passes.

Control check (no torch imports):
`rtk proxy python3 -m unittest discover -s research/engineering/radial_full_execution_v3 -p 'test_*.py' -v`

Portable build:
`rtk proxy python3 research/engineering/radial_full_execution_v3/build.py`

The output belongs to `kaggle/intensive_20261025_r1/`. Read START_HERE.md before any
execution. Launch ownership is coordinated by the portfolio owner; this chat
has not launched or scheduled a job. No commit, push or submission was performed.

Revision v3 adds the explicit frozen child-process environment and protects a 1,800-second resume allocation before full development. Recommended sessions: 1,800-second preflight, 24,000-second development, 1,800-second resume, each plus 900 export seconds. Total with prior charges is 64,394.580382916 GPU-seconds of the unchanged 64,800 cap. Reservation is not proof that any stage completes. The existing cost profile still estimates the whole matrix on resume and can refuse a short continuation; numerical recovery and remaining-work feasibility are unverified. Old packages remain untouched.
