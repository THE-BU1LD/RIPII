# RIPII 0.2 post-correction development protocol — DRAFT

Status: **superseded planning draft; never an evidential protocol**. The executable
development protocol is `post_correction_v02.md`.

## Purpose

Verify that the repaired projector-bank, graph, and VQ objectives are trainable and
directionally correct before spending compute on external confirmation. Historical
results remain attached to their original source and are not recomputed in place.

## Freeze requirements

Before execution, replace this section with immutable values for the source commit,
source hashes, dataset manifests, hardware, seed list, budgets, model grid, primary
metric, practical-effect threshold, checkpoint selection, failure rule, and output
location. Record an externally visible timestamp if the run will support a public claim.

## Development questions

1. Does projector count change capacity, gradients, and held-out reconstruction?
2. Does minimizing projection residual increase captured energy without representation
   collapse?
3. Does the differentiable edge-concentration term change attention entropy while the
   fixed top-k support and no-self-edge contract remain satisfied?
4. Does VQ balance improve effective code use, and does VQ outperform continuous or FSQ?
5. Does any hierarchy variant beat flat graph and global pool under matched information,
   capacity, updates, and measured compute?

## Advancement boundary

No positive method claim is allowed from this development study. Advancement to a
confirmatory protocol requires a consistent practically meaningful improvement on every
predeclared development domain, no material IID regression, healthy mechanism
diagnostics, and a compute-matched result. Failure on any required cell yields
`no_advance`; failed cells cannot be dropped.
