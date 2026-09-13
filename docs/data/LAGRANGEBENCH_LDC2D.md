# LagrangeBench 2D lid-driven cavity source card

- Source: LagrangeBench Datasets, Zenodo DOI `10.5281/zenodo.10491868`.
- Authors/publisher record: Artur P. Toshev and Nikolaus A. Adams, Technical
  University of Munich.
- License: CC BY 4.0 for the Zenodo dataset. The accompanying LagrangeBench code
  repository is MIT licensed; those are distinct grants.
- Pinned artifact: `2D_LDC_2708_10kevery100.zip`, 358,811,551 bytes,
  MD5 `f527ab73f69afc0d176a229f4b812cd5` as published by Zenodo.
- Retrieved SHA-256:
  `8e94edec374e1ba4ae59e56ac3a42b633016a8bd1f846cc2e1a7a201d63ce769`.
- Registry: `research/datasets/lagrangebench_ldc2d_v1.json`.

This is a genuine external community simulation benchmark, not empirical sensor data.
It contains roughly 2,708 fluid particles and therefore does **not** fit the current
small-object RIPII task without a scientifically material task/model adaptation. It must
not be silently truncated to eight objects or reported as a completed RIPII benchmark.

The integration gate is: verify the publisher artifact; preserve official train,
validation, and test boundaries; document coordinates, particle types, time units,
forcing, and normalization; implement scalable neighbor discovery; reproduce at least
one official baseline result; then freeze a task-specific comparison protocol before
evaluating RIPII. Normalization statistics must be fit on training data only.
