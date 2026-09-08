# Novelty audit

Reviewed through 2026-09-08. This is a targeted, primary-source desk audit, not a
systematic review.

## Verdict

No defensible novelty claim is established. RIPII combines established VAE/VQ,
message-passing, transformation conditioning, and learned coarse-to-fine processing.
The object-state workflow is a useful implementation and negative benchmark, but its
mathematical ingredients substantially overlap prior learned physics simulators and
multiscale graph networks.

## Closest established work

- Graph Networks as Learnable Physics Engines introduced object- and relation-centric
  graph dynamics for inference and control (Sanchez-Gonzalez et al., ICML 2018):
  https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html
- Graph Network-based Simulators established particle-state message passing and
  long-rollout physical simulation (Sanchez-Gonzalez et al., ICML 2020):
  https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
- MultiScale MeshGraphNets already uses fine/coarse message passing to address long
  graph distances (Fortunato et al., 2022): https://arxiv.org/abs/2210.00612
- Constraint-based Graph Network Simulator is a strong alternative formulation for
  learned physical dynamics (Rubanova et al., ICML 2022):
  https://proceedings.mlr.press/v162/rubanova22a
- VQ-VAE predates the repository's discrete latent mechanism (van den Oord et al.,
  2017): https://arxiv.org/abs/1711.00937
- Variational-RG/deep-learning connections predate the renormalization motivation
  (Mehta & Schwab, 2014): https://arxiv.org/abs/1410.3831

## Claim classification

| Category | Defensible status |
|---|---|
| New mathematics | Not established |
| New algorithm | Not established; learned pooling/unpooling overlaps multiscale GNNs |
| New combination | Possibly repository-specific, but combination novelty is unverified and unsupported by benefit |
| Benchmark | Internal synthetic simulator only; not a community benchmark |
| Systems | Reproducible negative-result capsule and fail-closed controls are useful engineering, not research novelty |
| Implementation | Implemented from scratch; this is not itself algorithmic novelty |

The very recent hierarchical learned-simulation literature further narrows the claim
space. No paper language should use “novel,” “state of the art,” or “significantly
better” without a broader systematic search and new evidence.
