# RIPII Audit Remediation Progress

This document tracks work against `END_TO_END_AUDIT_2026-09-13.md`. “Implemented”
means code or an artifact exists and has focused verification. It does not promote a
development result to confirmatory evidence.

## Overall status

- Weighted whole-audit completion: **approximately 69%**.
- Engineering/code-addressable work: **approximately 92%**.
- Publication-grade scientific validation: **approximately 35%**.
- Current scientific decision: **no advance / not conference-ready**.

The gap is dominated by a 180-cell objective study, powered external experiments,
third-party baseline integration, real data, independent reproduction, and human legal/
authorship decisions. Those cannot be truthfully replaced by code scaffolds.

## Completed or materially implemented

1. **Overnight reliability:** `scripts/run_post_correction.py` now uses repeated warmed
   calibration at three horizons, rejects timing inversions, uses a conservative upper
   bound, checks free space, runs a real 600-step sentinel, stores per-attempt logs,
   retries, quarantines partial output, resumes at seed-cell granularity, and verifies
   exact manifests. Status reports cell counts, last failure, PID liveness, and free
   space. A real 600-step sentinel completed and its status/manifest verified under
   `runs/audit_runner_sentinel_20260913/`.
2. **Complete evidence archive:** `scripts/archive_research_run.py` creates deterministic
   tarballs with an internal per-file manifest and sidecar digest. The 600-step sentinel
   was archived to `output/research/audit_runner_sentinel_20260913.tar.gz`.
3. **Physical evaluation:** rollout reports kinetic-energy RMSE, relative energy RMSE,
   contact-penetration error, wall penetration, passive-step momentum error, and action-
   free prevalence in addition to existing trajectory metrics.
4. **Symmetry evaluation:** every learned model reports numerical translation and
   quarter-turn equivariance errors rather than relying on architectural labels.
5. **Equivariant control:** an actual continuous E(2)-equivariant dynamics baseline was
   added and tested for translation, rotation, reflection-compatible construction, slot
   permutation, gradients, and checkpoint flow. A three-model executable smoke benchmark
   completed with 66/66 artifacts verified. It is not yet publication evidence.
6. **Sparse local messages:** local interaction layers gather only active neighbor edges
   for learned messages instead of expanding dense hidden pair tensors. Distance search
   remains quadratic; global variants necessarily remain all-pairs.
7. **Rollout robustness controls:** training supports recorded state noise and a
   deterministic horizon curriculum. Exact resume is tested with both enabled.
8. **Objective observability:** adaptive precision, effective weight, log variance, and a
   fixed-weight comparison total are now logged; invalid weights fail closed.
9. **VQ rescue mechanism:** deterministic explicit dead-code revival was added to the
   quantizer and qualification CLI. It never changes evaluation implicitly and requires
   a recorded schedule.
10. **NRI truth boundary:** dataset records now identify provenance for every field and
    mark property drift, action response, and relation recovery as inapplicable.
11. **Locked CI:** primary test and quality workflows now install and execute from
    `uv.lock` through a commit-pinned uv action.
12. **Paper artifact:** `scripts/build_paper.sh` now produces and validates
    `output/pdf/ripii-manuscript.pdf`. All three pages were rendered and visually checked.
    The paper now contains a result table, expanded references, ethics/disclosure boundary,
    and an explicit note that post-audit code has no frozen result.
13. **Cards and documentation:** added synthetic/NRI dataset cards, a world-model card,
    platform/determinism support, a current audit index, a license-status boundary, and a
    draft baseline-expansion protocol.
14. **Run catalogue and supply chain:** `scripts/index_runs.py` records complete, failed,
    incomplete, and unclassified run directories without calling inventory proof of
    validity. A CycloneDX 1.5 SBOM was exported from the lockfile.
15. **System-temp exhaustion:** the first full regression rerun exposed the system volume
    at 181 MiB free and failed fixtures/checkpoint writes after 139 passes. Pytest and
    subprocess temporary files now use ignored `.test-tmp/` on the research volume. The
    complete rerun then passed **170/170 tests** in 289.83 seconds.
16. **Type and coverage gates:** mypy found and drove fixes for a Python 3.10 datetime
    incompatibility, an incorrect return annotation, and ambiguous inferred container
    types. The complete `ripii` package now type-checks (41 files). Branch coverage is
    enforced at 75%; a fresh full run passed 176 tests at 80% total coverage.
17. **Performance/device gates:** CI now enforces a versioned CPU parameter/latency smoke
    contract and schedules a required MPS finite forward/backward repeatability check.
    CUDA has an opt-in self-hosted job; it remains uncertified until the owner provides a
    matching runner and enables the repository variable.
18. **Evidence-generated paper table:** `scripts/render_paper_results.py` verifies the
    immutable v3 capsule and renders the primary result paragraph/table into the paper.
    Tampering, a legacy capsule, a changed decision, or a missing/duplicate placeholder
    fails closed. The rebuilt three-page PDF was rendered and visually checked.
19. **Public benchmark acquisition:** the official LagrangeBench 2D LDC artifact (Zenodo
    `10.5281/zenodo.10491868`, CC BY 4.0) was downloaded, matched its published byte count
    and MD5, and received a local SHA-256. Official train/valid/test HDF5 files are intact.
    Its 2,708-particle SPH task is explicitly prohibited from being reported as a RIPII
    result until a scalable, task-valid adapter exists.
20. **Official baseline setup:** official LagrangeBench GNS/EGNN/SEGNN source is pinned at
    commit `b880a6c84a93792d2499d2a9b8ba3a077ddf44e2` and its 87-package stack installed in an
    isolated Python 3.11 environment after the authors' documented macOS PyTorch change.
    Package import then slept for more than 90 seconds at 0% CPU in dynamic-library
    loading and was terminated;
    no official score is claimed. GNS/MeshGraphNets and BSMS source revisions are pinned
    in `research/baselines/official_sources_v1.json`.
21. **Measured parallel overnight launch:** the frozen 180-cell matrix now executes the
    five independent seeds concurrently within each mode. Repeated five-seed batch
    calibration at 60/180/300 steps measured host contention directly and produced a
    conservative 8.57-hour estimate including a 25% safety factor. The 24-hour gate
    passed, a real 600-step sentinel cell completed and verified, and the resumable study
    is active in `runs/post_correction_v02_parallel_20260913/`. A closed foreground
    output pipe caused one preserved `BrokenPipeError` after 20 verified cells; no cell
    evidence was lost. The unchanged frozen run was resumed in the persistent tmux
    session `ripii-post-correction-20260913`, with five seed workers. This is execution
    in progress, not a result.
22. **Release fail-closed gate:** `scripts/check_release_readiness.py` verifies the PDF,
    SBOM, license, citation metadata, and owner approvals. It currently blocks exactly on
    `LICENSE`, `CITATION.cff`, and `release/OWNER_METADATA.json`; those values are not
    fabricated by automation.

## Partially addressed; more execution is required

1. **Full legacy-method viability:** the runner and diagnostics are fixed and one
   full-length cell passed and the remaining 179 frozen cells were launched under the
   measured five-worker schedule. Do not infer viability until the complete exact
   manifest and predefined decision are present.
2. **Strong baselines:** the E(2)-equivariant control is implemented. Official GNS,
   constraint-based GNS, MeshGraphNet/BSMS, and Lagrangian/energy baselines still need
   license review, adapters, fair tuning, and execution.
3. **Compute fairness:** existing parameter/profiling tools remain and the draft protocol
   requires equal tuning and compute. Publication-scale FLOP/wall-time/convergence matched
   runs do not yet exist.
4. **Physical architecture:** the new equivariant control and diagnostic tests close the
   symmetry comparison gap, but the historical generic model still uses unstandardized
   features and fixed residual scales. Changing them must be a new protocol, not a silent
   reinterpretation of old checkpoints.
5. **NRI semantics:** constructed fields are now machine-readable and documented. An
   edge-aware model/ablation and retained raw/checkpoint release are still absent.
6. **Paper reproducibility:** the primary v3 paragraph/table is now generated from its
   verified capsule by the PDF build. Other narrative result values still require the
   same treatment, and owner metadata is absent.
7. **Objective redesign:** logging and rescue controls are available, but no new objective
   has passed a locked development and confirmatory gate.
8. **Scaling:** sparse local messages reduce hidden-edge work, but neighbor discovery is
   still quadratic and no large-N scaling experiment has been completed.
9. **CI depth:** lock, package types, coverage, CPU performance, and scheduled MPS gates
   are configured. A real CUDA runner/result, device tolerance study, and first passing
   scheduled MPS run remain external execution requirements.
10. **Release:** full archives and an SBOM can be generated, but no clean tagged/DOI release
    or independent clean-machine replay has been completed.

## Blocked on external or human authority

1. **License and ownership:** only the owner can choose the software/data license and
   confirm copyright. `LICENSE_STATUS.md` prevents accidental implication of rights.
2. **Authorship/disclosures:** names, affiliations, contributions, conflicts, and funding
   require authorized human input.
3. **Empirical data:** a licensed genuine external community simulation benchmark is now
   pinned and locally verified, but it is not empirical sensor data. A scientifically
   valid scalable adapter and a separate empirical/noisy trajectory source remain needed;
   synthetic substitution would not close that part of the finding.
4. **Powered confirmation:** the frozen external study requires substantial compute and
   untouched test data. Current three-/five-seed results remain descriptive.
5. **Independent reproduction:** must be performed by another person/team from a clean
   release archive.

## Immediate next execution order

1. Monitor `runs/post_correction_v02_parallel_20260913/` to completion, verify its exact
   manifests, apply the predefined gate, and publish the complete archive; stop
   advancement if the gate fails.
2. Obtain owner license/authorship decisions and make the release gate pass.
3. Integrate official strong-baseline implementations only after license/API review.
4. Freeze the baseline-expansion protocol, run equal-budget development studies, and set
   power from blinded external-development variance.
5. Acquire and checksum a genuine external dataset, then run the untouched confirmatory
   protocol.
6. Build a clean tagged release, regenerate PDF/SBOM/archives, and arrange independent
   reproduction.
