# RIPII component pilot v1: decision record

Status: **completed local pilot; no advancement**.

This analysis is derived from `summary.json`. It does not modify the 129 artifacts
listed in `manifest.json`, all of which pass SHA-256 and size verification. The
manifest SHA-256 is
`f3014d42829c4e7566c20556b59a0f3f1712999cf099fe20efd177e510441672`.

## Predefined decision

The protocol required both:

1. lower mean test reconstruction MSE for `base` than `no_structured`; and
2. base held-out probe accuracy no more than 0.05 below `no_structured`.

| Outcome | Base mean | `no_structured` mean | Base minus comparison | Gate |
|---|---:|---:|---:|---|
| Test reconstruction MSE (lower is better) | 0.229093 | 0.202576 | +0.026517 | Fail |
| Held-out probe accuracy (higher is better) | 0.350877 | 0.526316 | -0.175439 | Fail |
| Held-out structural probe accuracy | 0.578947 | 0.578947 | 0.000000 | Descriptive only |

**Decision: do not advance the current method on the basis of pilot v1.** The full
model failed both required gates on the bundled synthetic task.

## Descriptive findings

`no_vq` had the lowest mean reconstruction MSE (0.184733), which is evidence against
the current quantizer helping reconstruction at this budget. The base model's mean
coarse and fine perplexities were 1.076 and 1.312 from codebooks of size eight, with
mean usage 0.1875. That is near-collapse, not successful discrete representation
learning. `no_graph` worsened reconstruction relative to base but improved the
held-out probe; this mixed result does not support a general graph benefit.

The held-out set contains 19 examples per seed, so probe accuracy moves in increments
of 1/19. Three seeds, one synthetic generator, and non-parameter-matched removals do
not support significance, generalization, or superiority claims.

## Post-run validity finding

Review after the run found that `AdaptiveLossBalancer` added a learned `log_var`
offset even when an ablation set the associated loss weight to zero. Those inactive
parameters received gradients and participated in global gradient clipping. This
makes the generated `summary.md` ranking by total loss invalid and weakens causal
interpretation of all component comparisons. A separate configuration defect also
made the historical `rec` key implicit rather than the canonical `recon`; its value
was 1.0, matching the fallback, so it did not numerically change this pilot.

The primary no-advance conclusion is deliberately retained: it is conservative, both
predefined gates failed, and no positive effect is claimed. Pilot v1 must not be cited
as a clean ablation or publication result. The fixes apply only to future runs.
