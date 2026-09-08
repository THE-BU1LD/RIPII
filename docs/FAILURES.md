# Failures and negative results

Pilot v1 exposed a loss-balancing defect. Pilot v2 showed near-collapsed VQ and worse
reconstruction. World v3 showed learned hierarchy losing the flat graph on all five
aggregate OOD seed comparisons. Failure localization did not isolate the problem to
contacts. A targeted long-range force made global context useful but did not make
learned grouping better than global pooling.

Plausible explanations are unnecessary optimization burden, non-identifiable soft
groups, and lack of task benefit beyond generic global aggregation. These are empirical
explanations, not proved causes. External datasets and compute matching are required
before generalizing the negative result.
