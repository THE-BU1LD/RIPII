# RIPII corrected component pilot v2: decision record

Status: **completed local pilot; no advancement**.

This record is derived from `summary.json`. All 51 files declared by `manifest.json`
pass SHA-256 and size verification, including the three exact paired initialization
checkpoints. The manifest SHA-256 is
`dfe6f6924216489d4e37f6e01f37dabbc3574094510e03f47511dc8ae3a1af10`.

## Frozen decisions

| Comparison/gate | Base | Comparison | Difference | Result |
|---|---:|---:|---:|---|
| Reconstruction vs `no_vq` (lower is better) | 0.297911 | 0.236362 | +0.061549 | Fail |
| Probe vs `no_vq` (base no more than 0.05 lower) | 0.491228 | 0.508772 | -0.017544 | Pass |
| Base coarse effective-code fraction (>0.25) | 0.147782 | — | — | Fail |
| Base fine effective-code fraction (>0.25) | 0.181503 | — | — | Fail |
| Reconstruction vs `no_structured` | 0.297911 | 0.236221 | +0.061691 | Fail |
| Probe vs `no_structured` (base no more than 0.05 lower) | 0.491228 | 0.578947 | -0.087719 | Fail |

**Decision: neither the quantizer nor the full mechanism stack earns further work in
its current form.** Only one of the four quantizer conditions passed, and neither
full-stack condition passed.

## Failure analysis

Quantizer bypass reconstructed better on every seed. Two of three base runs used
effectively one code in each eight-entry codebook; the remaining run reached
perplexities 1.547 coarse and 2.356 fine. This is not merely aggregate noise: the
discrete bottleneck is collapsing at the tested budget.

The full stack also reconstructed worse than `no_structured` on every seed. The
`no_structured` comparison is not a plain autoencoder: it retains the shared encoder,
node/fusion scaffold, and decoder and has fewer trainable parameters. Accordingly,
this result rejects advancement but does not identify which remaining scaffold is a
good general baseline.

The held-out set has 19 examples per seed, the study has three seeds, and all data
come from one bundled synthetic generator. No significance, external-generalization,
novelty, or superiority conclusion is authorized.

## Engineering checks demonstrated by v2

Zero-weight balanced terms are now exactly zero, adaptive `balanced_total` equals
reported `total`, disabled-quantizer diagnostics are null rather than fake perfect
usage, and exact paired initial states are retained. These repair the experiment
mechanics; they do not rescue the hypothesis.
