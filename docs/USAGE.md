# Using RIPII

RIPII 0.2 exposes its object-state dynamics model as the supported inference product.
The older structured-latent autoencoder remains a research surface and is not presented
as a general-purpose predictor.

## State contract

Each scene is padded to the checkpoint's `max_objects` value. Inputs are:

- `state`: float tensor `[batch, objects, 6]` containing `x, y, vx, vy, radius, mass`;
- `action`: float tensor `[batch, objects, 2]` containing `force_x, force_y`;
- `actions`: float tensor `[batch, time, objects, 2]` for a rollout;
- `mask`: boolean tensor `[batch, objects]`, true only for live objects.

Padded state and action entries must be zero. Every live radius and mass must be
positive. Position and time units must match the dataset recorded in the checkpoint.

## Python API

```python
import torch
from ripii.world import WorldPredictor

predictor = WorldPredictor.from_checkpoint("best.pt", device="cpu")
next_state = predictor.predict(state, action, mask)
trajectory = predictor.rollout(state, actions, mask)
metadata = predictor.inspect()
```

The returned rollout includes the initial state, so a request with `T` actions returns
`T + 1` states. `WorldPredictor` uses inference mode and moves inputs to the requested
device. CUDA requests fail rather than silently falling back to CPU.

Only load checkpoints from trusted sources. PyTorch's restricted loader and RIPII's
schema validation reduce risk but do not establish provenance.

## Command line

Install the package and inspect a checkpoint:

```bash
ripii inspect best.pt
```

Run one-step prediction from a non-pickled NumPy archive:

```bash
ripii predict best.pt --input step.npz --output prediction.npz
```

`step.npz` must contain exactly `state`, `action`, and `mask`. For a rollout:

```bash
ripii rollout best.pt --input rollout.npz --output trajectory.npz
```

`rollout.npz` must contain exactly `state`, `actions`, and `mask`. Both commands write
an archive containing `states`. Use `--device cuda` only on a CUDA-capable host.

## Training and evaluation

The existing experiment commands remain available under the same executable:

```bash
ripii train --output runs/my_model --steps 300 --model graph
ripii benchmark --output runs/my_benchmark --steps 300 --seeds 3 7 11
ripii verify runs/my_benchmark
```

Use `--model equivariant` for the continuous E(2)-equivariant baseline. Use
`--rollout-curriculum-steps N` to grow the training horizon to `--rollout-steps` over
the first N updates, and `--state-noise-std S` for an explicitly recorded state-noise
rollout intervention. Both default off so historical behavior is unchanged.

These commands produce development evidence. They do not turn a run into a
confirmatory experiment; that requires a protocol frozen before test evaluation.

## External trajectory datasets

RIPII accepts a directory containing `manifest.json` plus non-pickled NPZ files for at
least `train`, `validation`, and `test`. Verify the directory before use:

```bash
ripii verify-dataset datasets/example
ripii train --dataset-root datasets/example --output runs/example_graph --model graph \
  --rollout-steps 4
ripii evaluate runs/example_graph/best.pt --dataset-root datasets/example \
  --split test --output runs/example_graph/test.json
ripii benchmark-dataset --dataset-root datasets/example \
  --output runs/example_benchmark --steps 300 --seeds 3 7 11
```

Each NPZ contains exactly `states`, `actions`, `mask`, and `ids`. The first two arrays
follow the state contract above; `ids` is an `int64` trajectory identifier and must not
overlap between splits. The manifest uses this schema:

```json
{
  "format": "ripii-trajectory-dataset-v1",
  "dataset_id": "organization.dataset",
  "version": "immutable-version",
  "license": "SPDX-or-NOASSERTION",
  "units": "explicit physical units for every feature",
  "preprocessing": "deterministic preprocessing description",
  "split_policy": "how leakage is prevented",
  "observation_dt": 0.05,
  "artifacts": {
    "train": {"path": "train.npz", "bytes": 123, "sha256": "..."},
    "validation": {"path": "validation.npz", "bytes": 123, "sha256": "..."},
    "test": {"path": "test.npz", "bytes": 123, "sha256": "..."}
  }
}
```

Paths are restricted to the dataset directory, symlinks are rejected, hashes and sizes
are checked, object masks and physical properties are validated, and NumPy pickle
loading is disabled. The loader records license metadata but does not decide whether
redistribution is legally permitted.
