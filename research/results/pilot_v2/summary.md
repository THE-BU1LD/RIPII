# RIPII Benchmark Summary

Adaptive multi-objective totals are optimization diagnostics, not a
cross-ablation ranking metric. Compare outcomes defined by the study protocol.

## Descriptive outcome metrics

| mode | recon | heldout_probe_accuracy | heldout_structural_probe_accuracy | perplexity_coarse | perplexity_fine | usage | balanced_depth | balanced_equiv |
|---|---|---|---|---|---|---|---|---|
| base | 0.297911 | 0.491228 | 0.701754 | 1.182253 | 1.452026 | 0.187500 | -0.024466 | 0.420545 |
| no_structured | 0.236221 | 0.578947 | 0.666667 | nan | nan | nan | 0.000000 | 0.000000 |
| no_vq | 0.236362 | 0.508772 | 0.631579 | nan | nan | nan | -0.023625 | 0.432111 |

## Delta vs base

- no_vq: recon=-0.061549, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=-0.070175, balanced_depth=+0.000841, balanced_equiv=+0.011565, balanced_geom=-0.004241
- no_structured: recon=-0.061691, heldout_probe_accuracy=+0.087719, heldout_structural_probe_accuracy=-0.035088, balanced_depth=+0.024466, balanced_equiv=-0.420545, balanced_geom=-0.044082
