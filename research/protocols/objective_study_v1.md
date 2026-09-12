# Legacy objective study v1

Status: frozen local development protocol, written before executing this study on
2026-09-09. It is not externally timestamped and is not confirmatory evidence.

## Question and hypotheses

Does any one legacy auxiliary loss, or the complete 13-term objective, improve
held-out reconstruction over the simpler reconstruction-plus-KL objective at the same
fixed training budget?

- H0: no tested addition reduces held-out reconstruction MSE by at least 5% on every
  paired seed relative to reconstruction+KL.
- H1(term): adding `term` reduces held-out reconstruction MSE by at least 5% on every
  paired seed. The same rule applies to the complete objective.

This deliberately tests optimization behavior, not whether an internal diagnostic
matches its own regularizer.

## Frozen design

- Base configuration: `configs/pilot_v2.yaml`.
- Fresh paired seeds: `701`, `809`, `907`.
- Fixed budget: 30 optimizer updates; final checkpoint only.
- Shared data, split, initialization, batch order, architecture, and parameter count
  within each seed.
- Reference: `simple_objective` = reconstruction weight 1.0 and KL weight 0.01; all
  eleven auxiliary terms have exactly zero weight.
- Comparisons: `simple_plus_equiv`, `simple_plus_inv`, `simple_plus_scale`,
  `simple_plus_proj`, `simple_plus_spectral`, `simple_plus_geom`, `simple_plus_vq`,
  `simple_plus_node`, `simple_plus_moment`, `simple_plus_identity`,
  `simple_plus_depth`, and `base` (the complete configured objective).
- Primary outcome: held-out test reconstruction MSE, evaluated after the fixed budget.
- Secondary outcomes: held-out ridge-probe accuracy, representation diagnostics, and
  codebook utilization. They cannot override the primary decision.
- Experimental unit: paired initialization/minibatch seed.

## Decision and failure rules

An addition is provisionally supported only if it lowers reconstruction MSE by at
least 5% relative to `simple_objective` on all three paired seeds. Three seeds have
low inferential resolution; results are descriptive development evidence only.

Any missing cell, failed command, non-finite value, source/config/protocol hash
mismatch, absent paired initialization, overwritten artifact, or test-split reuse for
training invalidates the whole study. No failed seed may be omitted. This protocol
must be superseded—not edited—after results are observed.

## Canonical command

The exact command, including this file's SHA-256, is recorded beside the generated
summary after the digest is computed. Full checkpoints remain ignored local evidence;
the summary, raw seed rows, manifest, and protocol are retained.
