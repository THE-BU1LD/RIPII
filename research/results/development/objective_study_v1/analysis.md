# Objective study v1 analysis

Decision: **no_auxiliary_advance**.

| comparison | mean candidate − simple MSE | seed reductions | passes |
|---|---:|---|---|
| `base` | 0.100741 | -26.35%, -59.95%, -89.51% | False |
| `simple_plus_depth` | 0.002599 | -0.94%, -1.07%, -2.47% | False |
| `simple_plus_equiv` | 0.014403 | -2.62%, -6.24%, -15.88% | False |
| `simple_plus_geom` | 0.001318 | -1.00%, -0.52%, -0.78% | False |
| `simple_plus_identity` | 0.002330 | -0.79%, -1.07%, -2.17% | False |
| `simple_plus_inv` | 0.105956 | -30.29%, -58.45%, -95.63% | False |
| `simple_plus_moment` | 0.002167 | -0.70%, -0.99%, -2.06% | False |
| `simple_plus_node` | 0.004225 | -1.93%, -2.08%, -3.34% | False |
| `simple_plus_proj` | 0.007886 | -2.84%, -2.17%, -8.42% | False |
| `simple_plus_scale` | 0.008391 | -1.28%, -3.00%, -10.04% | False |
| `simple_plus_spectral` | 0.031589 | -9.60%, -16.19%, -29.03% | False |
| `simple_plus_vq` | 0.007011 | -5.57%, -4.17%, -2.65% | False |

Prefer reconstruction+KL for this fixed-budget synthetic setting; three seeds do not establish population noninferiority or external utility.

Positive reduction means the addition helped; negative means it hurt.
