# RIPII execution checklist

Updated: 2026-09-08. A checked item has executable local evidence. Historical pilots
remain immutable and are not upgraded by later code changes.

## P0 — correctness and artifact integrity

- [x] Fix disabled-objective uncertainty offsets and preserve the original failed
  pilot rather than rewriting it.
- [x] Repair weighted evaluation, identity transforms, exact resume state, and
  fail-closed configuration/numerical validation.
- [x] Replace the structured pseudo-baseline with a genuinely plain autoencoder and
  match trainable capacity without unused padding.
- [x] Verify both retained pilot manifests while reporting source drift explicitly.
- [x] Snapshot world-model source and hash every generated benchmark artifact.
- [x] Fix relative-path world-manifest verification and test the regression.
- [x] Reject empty, duplicate, unknown, boolean, and negative benchmark-grid inputs
  before creating an output directory.

## P1 — individual world-model controls

- [x] Implement MLP, graph, Transformer, global-pool, and multiscale predictors.
- [x] Test forward/backward behavior, masking, preserved physical properties, and
  permutation equivariance where the architecture promises it.
- [x] Add persistence, constant-velocity, and force/mass kinematic references.
- [x] Add multiscale-assignment, FSQ-utilization, and VQ-usage diagnostics.
- [x] Run a fresh five-model, three-seed control; retain its `no_advance` decision.
- [x] Add paired multiscale-versus-control diagnostics without converting them into
  unsupported significance claims.
- [x] Complete the fresh-data, five-seed, 1,000-update graph/global-pool/multiscale
  convergence study and verify its 124-file manifest. The predeclared result is
  `no_advance`; multiscale loses the graph OOD comparison on all five seeds.
- [x] Capture the ignored 55 MB full run as a compact signed research capsule that
  retains its protocol, complete summary, human-readable report, and full manifest.

## P1 — product and release path

- [x] Provide train, benchmark, verify, exact-resume, and intervention-demo commands.
- [x] Build wheel and sdist artifacts and exercise the installed world-model command
  outside the source checkout.
- [x] Run lint, compilation, lockfile, tests, retained-pilot verification, benchmark
  verification, and headless demo rendering.
- [x] Attach local `main` to the public source remote at commit
  `0dba137d49e9f827c6ca48724cb9f2106c81daf4`.
- [x] Verify portable retained-pilot summaries from a clean checkout without
  pretending the ignored full-run files are present.
- [ ] Add an owner-approved license and authorship metadata. Blocked on owner/legal
  choice; no license is inferred automatically.
- [x] Verify hosted CI for commit `0dba137d49e9f827c6ca48724cb9f2106c81daf4`:
  [CI run 34230280332](https://github.com/THE-BU1LD/RIPII/actions/runs/34230280332)
  and [quality run 34230280341](https://github.com/THE-BU1LD/RIPII/actions/runs/34230280341)
  both completed successfully.

## External scientific gates

- [ ] External real-world dynamics data or a learned perception front end.
- [ ] Compute-matched scaling study on independent datasets.
- [ ] Independent-human reproduction, expert novelty review, and peer review.

## Verification commands

```bash
ruff check .
python -m compileall -q ripii scripts tests
pytest
uv build
python scripts/verify_artifact.py --manifest research/results/pilot_v1/manifest.json --protocol research/protocols/pilot_v1.md --portable-summary
python scripts/verify_artifact.py --manifest research/results/pilot_v2/manifest.json --protocol research/protocols/pilot_v2.md --portable-summary
python -m ripii.world verify runs/world_v2_controls
python -m ripii.world verify runs/world_v3_convergence
python -m ripii.world verify-capsule research/results/development/world_v3_convergence_capsule_v2.json
```
