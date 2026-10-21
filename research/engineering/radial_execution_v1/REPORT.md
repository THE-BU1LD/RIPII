# Radial execution v1: qualified engineering optimization

The corrected candidate passed all five model equivalence fixtures on free Kaggle. The full scientific comparison remains cost-blocked and adaptive benefit remains unknown.

| Model | Initial update seconds | Captured update seconds | Descriptive timing ratio |
|---|---:|---:|---:|
| radial_mixture | 0.5350 | 0.0537 | 9.96 |
| radial_fixed | 0.4700 | 0.0482 | 9.75 |
| graph | 0.9993 | 0.0600 | 16.66 |
| simple_pool | 0.9588 | 0.0621 | 15.43 |
| transformer | 0.7543 | 0.0566 | 13.32 |

These are separate-run fixture maxima, not a paired performance estimate. The first profile included host data selection and scalar transfers; the captured measurement excludes data loading/copy. Clock and fixture state differ. No end-to-end speedup is established.

The corrected fixture uses AdamW weight decay 0.0001, verifies changing the device-tensor learning rate during replay, compares predictions and gradients at horizons 1/8/64 after nonzero head perturbations, tests radius-threshold and empty-edge cases, rejects NaN inputs, and requires exact serialization replay. Original and optimized fixtures are finite tests rather than universal proofs. Captured training only is implemented and qualified; integration into the full convergence-aware supervisor remains unimplemented and unqualified.

The minimum update-only extrapolation is 13.10 host-hours before validation, simulator generation, graph capture and safety reserve. Current remaining authorization is 3.4252 allocated GPU-hours, equivalent to 1.7126 hours on dual T4s. A full fit matrix is not admitted.

Attempt 1 used default AdamW decay and remains retained. Attempt 2 corrected this fixture mismatch without changing tolerances. No test splits, scientific training fits, or superiority results were produced. The original frozen radial study and historical v3 result were not edited.

## Reproduce

Use the exact retained Kaggle notebooks and their SOURCE manifests. `probe.py` requires CUDA and never falls back to local CPU. `build.py` creates a new reviewed package and explicitly carries the previous charge. Do not launch an old notebook to reset accounting. The independent standard-library intake command is:

```sh
python scripts/intake_radial_execution.py ARCHIVE NEW_DEST EXPECTED_PRELAUNCH_SOURCE_JSON
```

The current source manifest is under `portable/SOURCE.json`; each downloaded attempt also retains its original SOURCE manifest. Prior failed or inadequate attempts must remain in the cumulative charge ledger.

## Evidence

- [Corrected attempt intake](hosted/attempt-2/intake.json)
- [All model checks](hosted/attempt-2/extracted/results/probe.json)
- [Cumulative charge](hosted/attempt-2/extracted/charge.json)
- [State](STATE.json)
- [Private Kaggle run](https://www.kaggle.com/code/ryangomez010/ripii-radial-execution-v1-a9bc3d2b1442)
