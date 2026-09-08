# RIPII Benchmark Summary

## Ranking by total loss

- no_structured: 0.014887
- no_vq: 1.300505
- no_renorm: 3.053666
- no_action: 3.212398
- base: 3.752723
- no_graph: 3.753791

| mode | balanced_depth | balanced_equiv | balanced_geom | balanced_identity | balanced_inv | balanced_kl | balanced_moment | balanced_node |
|---|---|---|---|---|---|---|---|---|
| base | -0.017502 | 0.479725 | 0.058270 | -0.018867 | 0.111067 | -0.006618 | 0.024632 | 0.143503 |
| no_action | -0.017445 | -0.019159 | 0.056112 | -0.019159 | 0.126865 | -0.007421 | 0.024209 | 0.144711 |
| no_graph | -0.017325 | 0.473981 | 0.047308 | -0.018688 | 0.153082 | -0.004440 | 0.026084 | -0.019025 |
| no_renorm | -0.020245 | 0.471932 | -0.020245 | -0.019872 | 0.199135 | -0.007011 | 0.022829 | 0.145295 |
| no_structured | -0.029888 | -0.029888 | -0.029888 | -0.029888 | 0.066353 | -0.020453 | 0.014191 | -0.029888 |
| no_vq | -0.027619 | 0.388366 | 0.041339 | -0.028882 | 0.053357 | -0.017138 | 0.016114 | 0.141412 |

## Delta vs base

- no_renorm: balanced_depth=-0.002743, balanced_equiv=-0.007793, balanced_geom=-0.078515, balanced_identity=-0.001006, balanced_inv=+0.088068, balanced_kl=-0.000392
- no_graph: balanced_depth=+0.000177, balanced_equiv=-0.005744, balanced_geom=-0.010961, balanced_identity=+0.000179, balanced_inv=+0.042015, balanced_kl=+0.002178
- no_vq: balanced_depth=-0.010116, balanced_equiv=-0.091359, balanced_geom=-0.016931, balanced_identity=-0.010015, balanced_inv=-0.057709, balanced_kl=-0.010520
- no_action: balanced_depth=+0.000057, balanced_equiv=-0.498884, balanced_geom=-0.002157, balanced_identity=-0.000293, balanced_inv=+0.015798, balanced_kl=-0.000803
- no_structured: balanced_depth=-0.012385, balanced_equiv=-0.509613, balanced_geom=-0.088157, balanced_identity=-0.011021, balanced_inv=-0.044714, balanced_kl=-0.013835
