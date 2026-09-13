# Baseline Expansion v1 - Draft Protocol

Status: **draft; not preregistered; no confirmatory data may be inspected under this
document**.

## Primary question

Does multiscale grouping improve mean OOD position RMSE over a geometry-respecting
E(2)-equivariant interaction baseline when tuning budget, training compute, dataset
seeds, and model seeds are paired?

## Required design before freezing

- Models: multiscale, graph, global-pool, equivariant, an official noise-trained GNS
  implementation, a constraint-projected simulator, and an established multiscale
  simulator where the data representation is compatible.
- At least two independently generated development data seeds and one untouched
  confirmatory data-seed domain.
- Equal hyperparameter-trial counts; report parameters, recognized FLOPs, peak memory,
  training seconds, updates, and learning curves.
- Primary estimand: paired relative change in mean OOD position RMSE versus the strongest
  non-hierarchical baseline selected on validation only.
- Secondary metrics: velocity RMSE, kinetic-energy error, contact penetration, passive
  momentum error, symmetry errors, wall penetration, stability by horizon, and
  collision/force/speed strata.
- Seed count must be set from a blinded external-development pilot and may increase, but
  never decrease, the current 18-pair planning floor.
- Exact paired randomization inference, effect-size intervals, and one predeclared Holm
  family. All exclusions and failed runs remain visible.

## Advancement rule

Hierarchy advances only if it clears a predeclared practically meaningful improvement
on the untouched primary endpoint, has a compatible uncertainty interval, does not
materially worsen physical/symmetry diagnostics, and does not exceed the matched compute
budget. Code availability or assignment activity alone is not success.

