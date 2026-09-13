# Post-audit remediation status

Status: **engineering remediation in progress; scientific advancement not established**.

## Completed prospectively for 0.2

- `num_projectors` now controls a learned projector bank and selector capacity.
- Raw projector bases receive a nontrivial conditioning objective before QR.
- Projection loss minimizes unexplained residual energy instead of penalizing captured
  energy.
- Graph refinement excludes self-edges, reports actual support degree, and uses a
  differentiable attention-concentration objective instead of hard support as a loss.
- VQ uses differentiable soft-assignment balance; hard code usage is diagnostic only.
- Historical structured-latent checkpoints are rejected by corrected code rather than
  silently reinterpreted.
- The supported object-state product defaults to the empirically stronger graph model.
- `WorldPredictor` provides inspect, predict, and rollout APIs and non-pickled NPZ CLI
  interchange.
- Manifested external trajectory datasets can be verified, trained, evaluated, and
  benchmarked through public commands.
- Package metadata is alpha-scoped, CI covers Python 3.10–3.12, third-party actions are
  commit-pinned, and dependency/CodeQL security workflows are present.
- NRI negative evidence is integrated into the README, status, manuscript, and evidence
  ledger.

## Evidence boundary

These are prospective code corrections. They invalidate no historical negative result,
but historical scores do not evaluate the corrected implementation. New results require
a new source snapshot, dataset manifest, protocol, output directory, and artifact
manifest. The draft at `research/protocols/post_correction_v02_draft.md` is not frozen.

## Owner or external blockers

- Select the repository license and approve author, affiliation, contribution, conflict,
  and contact metadata.
- Select and legally review at least one fixed public observational dataset or a second
  separately maintained simulator family beyond NRI.
- Freeze a powered, compute-matched confirmatory protocol only after development results
  justify the expense.
- Run the complete paired seed grid on declared hardware and retain full checkpoints.
- Arrange independent clean-machine reproduction by someone outside development.
- Select a publication venue before producing venue-specific PDF and supplement files.

None of these blockers may be marked complete by adding prose or generated placeholder
artifacts. They require an owner decision, actual external data/compute, or an independent
person.

The synthetic post-correction matrix is now frozen in
`research/protocols/post_correction_v02.md` with a restart-safe, sleep-resistant
launcher. This closes the unattended execution gap, not the external blockers above.
