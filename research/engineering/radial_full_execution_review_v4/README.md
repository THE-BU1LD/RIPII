# RIPII frozen-reference qualification review v4

This is a prospective engineering derivative of the recovered executor v3. The 140-fit radial-adaptation-v2 comparison remains unexecuted. This package contains no efficacy result or executed GPU qualification.

## What changes

The predecessor preflight compared CUDA capture with an eager copy of the same optimized model. It could not detect a difference shared by both copies from the frozen scientific reference. The full supervisor did contain that reference check, but reached it only after accepting preflight and starting the full reservation. It also re-entered separate legacy capture machinery.

Preflight now checks original-versus-optimized predictions and active gradients at all seven curriculum horizons, optimizer updates at horizon 8, and serialized forecast replay at horizon 64 before capture and recovery. It retains nonzero-head fixtures, graph-edge cases and invalid-input rejection. Admission requires frozen_reference_equivalence=PASS and capture_lr_change=PASS. The LR check mutates the existing optimizer LR tensors after capture and compares the following replay with eager model and optimizer states.

The full supervisor uses probe.py --reference-only for a fresh reference check. The current CapturedEngine cost profile and four-lane smoke remain mandatory; the separate legacy probe.profile is excluded from that production path. Numerical thresholds, model, protocol, splits, fit count and budget are unchanged.

## Exact predecessor and preservation

The recovered 84-member v3 archive SHA-256 is 31694c6a7f604d7afae2866a930ce379a46f7900585532d031526bee0dfca8de; its inventory identity is f43e71379037622523200775366ed758de708e6fef2890d2147c0a8dc3606a20. BASELINE_SOURCE_V3.json retains its original hashes. All 63 reference files plus optimized.py, captured_engine.py, cost_profile.py and replay_checkpoint.py are byte-identical.

The existing CUBLAS environment and continuation-reservation corrections are reused. Their 28-test receipt is retained history. REVISION.json, VALIDATION_PLAN.md and HISTORICAL_README_V3.md describe that earlier scope. This README and CURRENT_REVISION.json govern the present review.

## Reproduce the scoped engineering checks

From the repository root:

```bash
python3 research/engineering/radial_full_execution_review_v4/build_review.py --output output/ripii-reference-review-v4
python3 -m unittest discover -s research/engineering/radial_full_execution_review_v4 -p 'test_*.py' -v
python3 research/engineering/radial_full_execution_review_v4/validate_package.py output/ripii-reference-review-v4
```

The builder verifies frozen bytes, parses Python and creates a deterministic source ZIP and embedded notebook. The complete portable reference source is included; no old workstation paths are required. Tests use the standard library. Six new methods test rejected legacy/missing/invalid receipts and reference-before-capture control flow with artificial child-process receipts. These are not numerical evidence. The current scoped suite has 34 unique methods: 28 retained and six new.

## Numerical execution remains unqualified

New numerical execution is restricted to private free Kaggle T4 x2, internet off and no paid scaling. Verify the intended account, live quota and existing jobs before launch. This review has no authenticated Kaggle session or CUDA device and launches no job.

The unchanged 18 allocated GPU-hour cap is 64,800 GPU-seconds, including 3,794.580382916 already charged. Retained sessions are 1,800 seconds for preflight, 24,000 for development and 1,800 for one continuation, each with 900 export seconds and both devices charged. The cumulative ceiling is 64,394.580382916 GPU-seconds. This arithmetic establishes neither live quota nor completion time. The stronger preflight must fit its existing reservation.

Keep the notebook in PREFLIGHT. Exact-source reference parity, capture parity, LR changes, gate intervention and fresh-process resume must pass for all five models before development admission. A fresh cost profile, reference check and four-lane smoke remain required in development. All scientific adequacy gates must pass before protected data opens. Independent confirmation remains refused because no such protocol exists.

CUDA equivalence, numerical recovery, dependencies, full-matrix feasibility, the 140 fits and efficacy remain unverified. Historical negative and undertrained results remain separate. This code review makes no conference-readiness or submission claim.
