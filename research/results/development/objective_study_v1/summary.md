# RIPII Benchmark Summary

Adaptive multi-objective totals are optimization diagnostics, not a
cross-ablation ranking metric. Compare outcomes defined by the study protocol.

## Descriptive outcome metrics

| mode | recon | heldout_probe_accuracy | heldout_structural_probe_accuracy | perplexity_coarse | perplexity_fine | usage | balanced_depth | balanced_equiv |
|---|---|---|---|---|---|---|---|---|
| base | 0.271419 | 0.473684 | 0.684211 | 1.408740 | 1.258583 | 0.208333 | -0.024202 | 0.397954 |
| simple_objective | 0.170678 | 0.491228 | 0.701754 | 2.385844 | 2.801069 | 0.416667 | 0.000000 | 0.000000 |
| simple_plus_depth | 0.173277 | 0.491228 | 0.684211 | 2.328692 | 2.861031 | 0.416667 | -0.023240 | 0.000000 |
| simple_plus_equiv | 0.185081 | 0.491228 | 0.666667 | 1.993428 | 1.751428 | 0.312500 | 0.000000 | 0.367708 |
| simple_plus_geom | 0.171996 | 0.543860 | 0.666667 | 2.683967 | 2.827682 | 0.437500 | 0.000000 | 0.000000 |
| simple_plus_identity | 0.173008 | 0.491228 | 0.666667 | 2.375119 | 2.861031 | 0.416667 | 0.000000 | 0.000000 |
| simple_plus_inv | 0.276634 | 0.438596 | 0.701754 | 1.619482 | 1.236764 | 0.229167 | 0.000000 | 0.000000 |
| simple_plus_moment | 0.172845 | 0.491228 | 0.684211 | 2.573297 | 2.661972 | 0.437500 | 0.000000 | 0.000000 |
| simple_plus_node | 0.174903 | 0.631579 | 0.701754 | 2.639288 | 2.801069 | 0.416667 | 0.000000 | 0.000000 |
| simple_plus_proj | 0.178564 | 0.508772 | 0.719298 | 2.277972 | 2.641867 | 0.375000 | 0.000000 | 0.000000 |
| simple_plus_scale | 0.179069 | 0.491228 | 0.701754 | 1.778629 | 2.196205 | 0.333333 | 0.000000 | 0.000000 |
| simple_plus_spectral | 0.202267 | 0.473684 | 0.561404 | 2.176717 | 2.257725 | 0.354167 | 0.000000 | 0.000000 |
| simple_plus_vq | 0.177689 | 0.631579 | 0.719298 | 2.635592 | 2.833244 | 0.520833 | 0.000000 | 0.000000 |

## Delta vs base

- simple_objective: recon=-0.100741, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=+0.017544, perplexity_coarse=+0.977104, perplexity_fine=+1.542486, usage=+0.208333
- simple_plus_equiv: recon=-0.086338, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=-0.017544, perplexity_coarse=+0.584688, perplexity_fine=+0.492845, usage=+0.104167
- simple_plus_inv: recon=+0.005215, heldout_probe_accuracy=-0.035088, heldout_structural_probe_accuracy=+0.017544, perplexity_coarse=+0.210742, perplexity_fine=-0.021819, usage=+0.020833
- simple_plus_scale: recon=-0.092350, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=+0.017544, perplexity_coarse=+0.369889, perplexity_fine=+0.937622, usage=+0.125000
- simple_plus_proj: recon=-0.092855, heldout_probe_accuracy=+0.035088, heldout_structural_probe_accuracy=+0.035088, perplexity_coarse=+0.869232, perplexity_fine=+1.383284, usage=+0.166667
- simple_plus_spectral: recon=-0.069152, heldout_probe_accuracy=+0.000000, heldout_structural_probe_accuracy=-0.122807, perplexity_coarse=+0.767976, perplexity_fine=+0.999141, usage=+0.145833
- simple_plus_geom: recon=-0.099423, heldout_probe_accuracy=+0.070175, heldout_structural_probe_accuracy=-0.017544, perplexity_coarse=+1.275227, perplexity_fine=+1.569099, usage=+0.229167
- simple_plus_vq: recon=-0.093730, heldout_probe_accuracy=+0.157895, heldout_structural_probe_accuracy=+0.035088, perplexity_coarse=+1.226852, perplexity_fine=+1.574661, usage=+0.312500
- simple_plus_node: recon=-0.096516, heldout_probe_accuracy=+0.157895, heldout_structural_probe_accuracy=+0.017544, perplexity_coarse=+1.230548, perplexity_fine=+1.542486, usage=+0.208333
- simple_plus_moment: recon=-0.098574, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=+0.000000, perplexity_coarse=+1.164557, perplexity_fine=+1.403388, usage=+0.229167
- simple_plus_identity: recon=-0.098411, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=-0.017544, perplexity_coarse=+0.966379, perplexity_fine=+1.602448, usage=+0.208333
- simple_plus_depth: recon=-0.098142, heldout_probe_accuracy=+0.017544, heldout_structural_probe_accuracy=+0.000000, perplexity_coarse=+0.919952, perplexity_fine=+1.602448, usage=+0.208333
