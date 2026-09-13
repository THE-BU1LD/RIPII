# Hierarchical quantizer qualification — 2026-09-13

Status: **failed development-only mechanism gate**

The existing frozen `scripts/qualify_quantizer.py` protocol was executed for
seeds 17–21 with 300 optimization steps, 512 training samples, 256 held-out
samples, a 16-dimensional code, and four coarse/four fine codes. No threshold
or implementation was changed after observing results.

| Seed | Coarse effective fraction | Fine effective fraction | Coarse purity | Fine purity | Reconstruction MSE | Qualified |
|---:|---:|---:|---:|---:|---:|---|
| 17 | 1.00 | 1.00 | 1.00 | 0.25 | 0.023902 | no |
| 18 | 1.00 | 1.00 | 1.00 | 0.25 | 0.023810 | no |
| 19 | 1.00 | 1.00 | 1.00 | 0.25 | 0.024048 | no |
| 20 | 1.00 | 1.00 | 1.00 | 0.25 | 0.023760 | no |
| 21 | 1.00 | 1.00 | 1.00 | 0.25 | 0.023979 | no |

All codes receive assignments and reconstruction is below the declared 0.1
gate, but the fine assignments are at the four-class chance rate in every
seed. Occupancy therefore does not establish hierarchical factor recovery.
This independently reinforces the retained codebook/hierarchy failure and
blocks promotion.

The next experiment must be a newly frozen intervention study. It should test
whether coarse-code absorption, greedy residual assignment, the balance term,
or decoder bypass causes the failure. Candidate remedies must be selected on a
development split and evaluated on fresh seeds; these five seeds cannot be
reused for confirmatory selection.
