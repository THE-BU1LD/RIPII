# Falsifiable hypotheses

## H1-WORLD-MULTISCALE (evaluated; development evidence)

- **H0:** learned coarse grouping gives no meaningful advantage over the local graph
  control under matched data, updates, initialization seeds, and approximately matched
  parameter count.
- **H1:** continuous multiscale reduces the mean of the three OOD position RMSEs by
  at least 5% versus continuous graph on every paired seed, while worsening IID test
  position RMSE by no more than 5% on every seed.
- Independent variable: processor (`multiscale` versus `graph`). Dependent variable:
  rollout position RMSE. Secondary: velocity RMSE, horizon errors, wall time,
  assignment diagnostics. Data: bundled simulator; train/validation/test/OOD policies
  in `WORLD_MODEL.md`. Seeds: 23, 29, 31, 37, 43 for the strongest study. Selection:
  minimum validation position RMSE + 0.25 velocity RMSE. Stopping: 1,000 updates.
- Decision: **H1 rejected by the predeclared development gate**; multiscale loses the
  graph OOD comparison on 5/5 seeds. This is not external population inference.

## H1-LEGACY-STRUCTURE (evaluated; frozen local pilot)

- **H0:** the projective/graph/VQ/action stack has no meaningful reconstruction or
  held-out probe advantage over its removal.
- **H1:** base has lower mean test reconstruction MSE than `no_structured` and probe
  accuracy no more than 0.05 lower, under the exact pilot v2 protocol.
- Decision: **H1 rejected**. The full stack reconstructs worse and codebooks are near
  collapse. See `research/protocols/pilot_v2.md`.

## Next confirmatory hypothesis (not yet run)

On at least two public physical-dynamics datasets with independently sampled scenes,
a local graph model will equal or outperform this learned grouping variant after
validation-only tuning and matched wall-clock/FLOP budgets. Primary endpoint: paired
per-scene long-rollout position error summarized first within dataset, then across at
least five training seeds. A practically meaningful difference is 5%. Exact model and
dataset choices must be frozen in a new versioned protocol before any test evaluation.

## Ranked extensions

1. **Simplify to graph:** tests whether hierarchy is unnecessary; lowest cost and
   directly supported by current evidence. Implemented as the recommended canonical
   model, with no claim of novelty.
2. **Geometry-aware coarse assignment:** assign from relative position plus state,
   targeting unstable semantic grouping. Control against fixed spatial bins and global
   pooling; falsified if it does not beat both under compute matching. Not implemented:
   current evidence does not justify more architecture before external benchmarking.
3. **Conservation-aware residual:** project predicted impulses to conserve pairwise
   momentum absent walls/drag. Control against the analytic force baseline; falsified
   if conservation improves diagnostics but not long rollouts. Moderate cost; not run.
4. **Noise-trained graph rollouts:** inject calibrated state noise to target rollout
   drift, with zero-noise control. Low novelty but high explanatory value; not run.

Extensions are ranked by falsifiability and explanatory value, not novelty theater.
