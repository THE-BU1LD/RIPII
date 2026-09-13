# RIPII 0.2 post-correction overnight protocol

Status: **frozen development protocol**. This protocol evaluates repaired mechanism
semantics. It is not confirmatory evidence and cannot establish novelty or superiority.

## Fixed design

- Dataset: repository synthetic paired-view benchmark, generated deterministically by
  each seed; 4,096 examples with the committed configuration.
- Seeds: 1009, 1013, 1019, 1021, 1031.
- Updates: 600 per mode-seed cell; batch size 64; validation every 20 updates.
- Modes: base; reconstruction+KL; each corrected auxiliary added independently;
  projector counts 1/2/4; hierarchy depths 0/1/2/3; graph top-k 1/2/full; VQ with and
  without balance; every registered component-removal ablation; and the
  capacity-matched plain autoencoder control (36 conditions, 180 cells).
- Primary development metric: held-out reconstruction MSE. All other reported metrics
  are diagnostics and cannot replace the primary metric after results are observed.
- Pairing: modes share seed, generated data, split policy, and compatible initial
  tensors. The plain architecture shares the seed but cannot share incompatible tensors.
- No failed or non-finite cell may be silently excluded. Interrupted mode groups may be
  rerun; completed groups must pass manifest verification and are never overwritten.
- Development gate: full base reconstruction MSE must be at least 5% lower for every
  paired seed than both reconstruction+KL and the capacity-matched plain autoencoder.
  Failure yields `no_advance`. Paired bootstrap intervals, exact sign-flip tests, and
  Holm-adjusted p-values are descriptive because five seeds are not confirmatory power.

## Integrity and decision boundary

At first launch, the runner records SHA-256 hashes for this protocol, the configuration,
all Python sources, `pyproject.toml`, and `uv.lock`. A resumed launch aborts on drift.
Every mode group receives its own raw rows, report, retained checkpoints, and manifest.

This is a five-seed synthetic development screen. Regardless of outcome, it does not
authorize a public method claim. Advancement requires the separately specified licensed
external datasets, prospectively powered paired seeds, compute matching, uncertainty
analysis, and independent reproduction.
