# RIPII research assessment source

Reviewed: 2026-09-07. This is a focused desk review of the mechanisms actually present
in RIPII, not a systematic literature review. Novelty remains unverified.

## Current interpretation

RIPII combines known families: variational autoencoding, vector quantization, learned
projection/coarse-graining, message passing, and transformation-conditioned latent
updates. The combination may be empirically useful, but the repository currently has
no evidence that these primitives or their combination constitute a new method.

The most diagnostic near-term question is not “is RIPII novel?” but “does any claimed
mechanism improve held-out behavior over a simpler model under a paired protocol?”
The pilot protocol is intentionally scoped to that question.

## Pilot v1 result

The 18-run frozen local pilot did not pass its advancement rule. Across three seeds,
base mean reconstruction MSE was 0.229093 versus 0.202576 for `no_structured`
(base minus comparison: +0.026517; lower is better). Mean held-out probe accuracy was
0.350877 versus 0.526316 (difference: -0.175439). Base codebook perplexity was near
collapse: 1.076 coarse and 1.312 fine from codebooks of size eight.

These results are descriptive and synthetic. After execution, review identified that
zero-weight objectives still contributed uncertainty offsets and gradients to the
adaptive loss balancer. The generated total-loss ranking is therefore invalid, and
global gradient clipping prevents treating this pilot as a clean causal ablation.
The preregistered no-advance decision is retained because both primary gates failed;
no positive component claim is made. A corrected protocol must use the prospective
fix and a simpler, explicitly specified baseline.

## Corrected pilot v2 result

Pilot v2 used three new seeds after fixing zero-weight objectives. Base mean
reconstruction MSE was 0.297911 versus 0.236362 for `no_vq` (difference +0.061549)
and 0.236221 for `no_structured` (difference +0.061691). Base held-out probe accuracy
was 0.491228, versus 0.508772 and 0.578947 respectively. Base effective-code fractions
were only 0.148 coarse and 0.182 fine. The quantizer failed its reconstruction and
both utilization gates; the full stack failed both gates against `no_structured`.
Every declared artifact and exact initial state passes manifest verification. This
is a credible negative local pilot, not evidence for RIPII.

## Primary sources and implications

- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
  establishes VQ-VAE-style discrete latent learning; hierarchical quantization alone
  is not a novelty claim.
- [SQ-VAE](https://proceedings.mlr.press/v162/takida22a.html) identifies codebook
  collapse as a central VQ failure mode. RIPII must report utilization and perplexity;
  its historical perplexity of one is a failure indicator, not success.
- [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html)
  provides the established message-passing framework. A learned latent graph requires
  task evidence beyond including a graph block.
- [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html)
  demonstrates architectural equivariance. RIPII currently uses an equivariance loss
  and transformation conditioning; it does not establish an exact group-equivariant
  architecture.
- [An exact mapping between the Variational Renormalization Group and Deep Learning](https://arxiv.org/abs/1410.3831)
  predates RIPII's coarse-graining motivation. “Renormalization” requires operational
  scale-flow evidence, not only stacked learned projections.
- [Soft Convex Quantization](https://proceedings.mlr.press/v242/gautam24a.html) reports
  strong reconstruction and utilization from an alternative quantizer, making a plain
  VQ implementation an insufficient eventual baseline.

## Evidence boundary

No source above validates RIPII. No claim of novelty, superiority, scalability, or
real-world utility is supported. The source set and negative pilot motivate stronger
baselines, quantizer repair, and failure diagnostics only.
