# Conference readiness checklist

Updated 2026-09-09. Evidence references are repository-relative. An unchecked item is
not silently satisfied by prose.

## A–F: question, mathematics, implementation, data, baselines

- [x] **A/P0 — understood.** State: two distinct research paths mapped. Evidence:
  `audit/REPOSITORY_MAP.md`. Change: separate claims/evidence. Verify: map reaches raw
  artifacts.
- [x] **B/P0 — falsifiable.** State: H0/H1 and effects defined. Evidence:
  `research/HYPOTHESES.md`. Change: replace vague hierarchy claim. Verify: decision rule
  precedes any new run.
- [x] **C/P1 — specified.** State: equations match code. Evidence:
  `research/MATHEMATICAL_SPEC.md`. Change: classify soft pooling, not physical RG.
  Verify: code-symbol inspection and mechanism tests.
- [x] **D/P0 — implemented.** State: all canonical forwards/backwards execute. Evidence:
  `tests/test_forward.py`, `tests/test_world.py`. Change: none required. Verify: pytest.
- [ ] **E/P1 — external data missing.** State: synthetic only. Evidence:
  `RESEARCH_TRUTH.md`. Required: two versioned public datasets with licenses/hashes.
  Verify: loaders and split tests. **EXTERNAL_EXECUTION_REQUIRED**.
- [x] **E/P0 — split isolation.** State: deterministic disjoint generators/indices.
  Evidence: `tests/test_research_boundaries.py`, `tests/test_world.py`. Verify: pytest.
- [x] **F/P1 — strong local controls.** State: analytic, MLP, graph, Transformer,
  global-pool, plain AE, removals. Evidence: `WORLD_MODEL.md`. Verify: benchmark report.
- [ ] **F/P1 — compute matching incomplete.** State: parameters and updates matched,
  not FLOPs. Required: predeclared compute budget. Verify: measured cost within 5%.

## G–M: training, evaluation, ablations, statistics, robustness, OOD, scaling

- [x] **G/P0 — validation selection and resume.** State: best validation checkpoint and
  exact CPU resume. Evidence: `tests/test_world.py`. Verify: uninterrupted/resumed states
  identical.
- [x] **H/P0 — fail-closed evaluation.** State: non-finite/empty-invalid inputs fail.
  Evidence: `tests/test_world.py`, `tests/test_research_boundaries.py`. Verify: pytest.
- [x] **I/P1 — real ablations.** State: legacy flags and world controls change execution.
  Evidence: ablation and model tests. Verify: gradients/outputs/modules differ.
- [x] **J/P1 — reusable statistics.** State: raw seed metrics, descriptive CI, exact
  paired sign-flip, Holm adjustment. Evidence: `ripii/utils/statistics.py`. Verify:
  `tests/test_statistics.py`.
- [x] **J/P1 — prospective power design.** State: current studies remain underpowered at
  3 and 5 seeds, but a self-checksummed development-only calculation fixes a 5% MDE, two-sided
  alpha 0.05, 80% target power, and recommends 18 paired seeds per external dataset.
  Evidence: `research/planning/external_power_plan_v1.json`. External pilot variance may
  raise, but must not lower, this count.
- [x] **K/P2 — bounded robustness.** State: more objects, composition, faster motion.
  Evidence: v3 capsule. Verify: capsule signature.
- [ ] **L/P1 — true external OOD missing.** State: same simulator law only. Required:
  independent datasets/simulators. Verify: dataset identities/hashes.
- [ ] **M/P2 — scaling missing.** Required: model/data/object-count curves and uncertainty.
  Verify: raw per-seed curves. **EXTERNAL_EXECUTION_REQUIRED**.

## N–Q: efficiency, reproducibility, tests, code quality

- [x] **N/P2 — partial efficiency.** State: parameters, wall time, inference latency,
  checkpoint size available. Evidence: world summaries/manifests. Verify: generated rows.
- [x] **N/P2 — profiling harness.** State: synchronized repeated measurements now report
  bootstrap timing intervals, throughput, checkpoint size, accelerator peak memory when
  available, environment, and a labeled recognized-operator FLOP lower bound. Evidence:
  `scripts/profile_world.py`. Comparative 5%-tolerance compute matching remains open in F.
- [x] **O/P0 — provenance.** State: configs, source hashes, versions, seeds, data seed,
  checkpoints, metrics, manifests. Evidence: `ripii/world/experiment.py`. Verify: verifier.
- [x] **O/P1 — canonical path.** State: preflight/test/smoke/experiment/analyze/paper/
  artifact scripts. Evidence: `scripts/*.sh`. Verify: commands in README.
- [x] **P/P0 — broad tests.** State: imports, shapes, gradients, flags, split leakage,
  resume, CLI, checkpoints, statistics, artifacts, failures. Evidence: `tests/`.
  Verify: full pytest.
- [x] **Q/P1 — canonical implementations.** State: no executable placeholder scan hits,
  no broad exception swallowing. Evidence: `audit/FINAL_AUDIT.md`. Verify: Ruff/compile.

## R–W: documentation, novelty, paper, release, external work

- [x] **R/P1 — documentation aligned.** State: README/status/world docs plus map/spec.
  Verify: links and commands.
- [x] **S/P1 — bounded related-work audit.** Evidence: `research/NOVELTY_AUDIT.md`.
  Verify: primary-source links.
- [x] **T/P0 — novelty claims constrained.** State: none established. Evidence: novelty
  audit. Verify: no unsupported novelty language.
- [x] **U/P1 — honest manuscript source.** State: negative-result draft only. Evidence:
  `paper/MANUSCRIPT.md`. Verify: every number maps to ledger.
- [ ] **U/P1 — submission paper unavailable.** Required: external/powered evidence,
  authorship, venue formatting. Verify: independent review and build.
- [x] **V/P1 — portable verification.** State: frozen summaries and self-checksummed capsule verify.
  Evidence: `scripts/verify_artifact.sh`. Verify: script exits zero.
- [ ] **V/P0 — license/authorship unresolved.** Required: owner-approved files. Verify:
  package metadata and legal review. This cannot be inferred by the maintainer.
- [ ] **W/P1 — external execution.** Run the draft external protocol only after dataset,
  license, compute, seeds and timestamp are fixed. Verify: immutable protocol digest.

## Mechanistic extensions ranked

1. **Simplify to flat graph (implemented control; strongest).** Hypothesis: hierarchy
   is unnecessary. Mechanism: remove grouping/coarse path. Benefit: less complexity;
   failure targeted: decorative groups. Control: matched graph. Falsified if hierarchy
   wins the frozen external protocol. Cost: low. Current evidence favors simplification.
2. **Spatially constrained grouping (not implemented).** Hypothesis: unconstrained soft
   groups mix nonlocal bodies. Formulation: penalize $S_{ig}S_{jg}$ for distant pairs.
   Control: equal-cost graph/global pool. Falsified if locality changes assignments but
   not OOD error. Cost: medium; unjustified until assignment-error analysis exists.
3. **Contact-event supervision (not implemented).** Hypothesis: sparse collision labels
   improve long-horizon contacts. Control: same labels supplied to graph. Falsified if
   contact-conditioned error is unchanged. Cost: medium and adds supervision, so it is
   not a clean rescue of the original hypothesis.
4. **Dynamic compute hierarchy (not implemented).** Hypothesis: conditional coarse
   updates improve efficiency. Control: equal-FLOP graph. Falsified if latency/FLOPs or
   error do not improve. Cost: high; current model executes all paths, so no such claim.
