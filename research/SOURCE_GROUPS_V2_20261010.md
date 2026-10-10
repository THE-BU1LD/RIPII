# External trajectory source groups, format v2

## Admission question and limitations

The existing external-data verifier detects shared row IDs and exact copied
trajectory content across splits. Different overlapping windows of the same
source episode need not have identical IDs or content. A future study therefore
needs a declared source-episode boundary as well as those existing checks.

The opt-in format `ripii-trajectory-dataset-v2` requires source-group metadata.
All windows assigned the same exact `(source_id, episode_id)` pair must belong
to one split. Multiple windows from that pair remain valid within a single
split. This is a declaration check. It does not authenticate source provenance,
resolve aliases, discover undeclared overlapping episodes, or establish
statistical independence. A future statistical analysis must choose independent
units from its scientific design; the loader does not choose them.

Valid v1 manifests remain supported. They receive no inferred group identity.
Duplicate JSON keys are now rejected for both versions because an ambiguous
manifest cannot identify one exact dataset declaration.

## Exact manifest contract

Retain the existing provenance, license, units, split policy and NPZ artifact
declarations. Change `format` to `ripii-trajectory-dataset-v2` and add
`source_groups` with exactly the same keys as the declared artifacts. For
example, the following illustrates only the additional metadata:

```json
{
  "format": "ripii-trajectory-dataset-v2",
  "source_groups": {
    "train": [
      {"trajectory_id": 1, "source_id": "source-A", "episode_id": "episode-1"},
      {"trajectory_id": 2, "source_id": "source-A", "episode_id": "episode-1"}
    ],
    "validation": [
      {"trajectory_id": 3, "source_id": "source-A", "episode_id": "episode-2"}
    ],
    "test": [
      {"trajectory_id": 4, "source_id": "source-B", "episode_id": "episode-1"}
    ]
  }
}
```

Each list must be nonempty. Every row must contain exactly `trajectory_id`,
`source_id` and `episode_id`. Trajectory IDs are signed int64 values, excluding
booleans, and are globally unique across declarations. Source and episode
identifiers are exact printable strings of one through 256 characters without
leading or trailing whitespace. They are not normalized or inferred.

The loader first admits the complete manifest and checks every source group,
including any additional OOD split, before opening an NPZ archive. A declared
pair cannot appear in two splits. When a split is loaded, its NPZ row IDs must
match that split's declared IDs one-to-one, with no missing or extra rows. The
existing strict container, source-byte hash, tensor-shape, finite-value and
full-verifier copied-content checks still apply.

The loaded record contains the actual format, row-aligned group records,
canonical group SHA-256 values, the number of unique declared groups and a
digest of the sorted unique group cohort. These are identities of declarations,
not independent evidence of how the source was collected. Metadata processing
uses linear storage in declared rows; sorted cohort hashing adds the usual
sorting cost. Tensor loading and exact-content checks keep their prior roles.

## Bind verification to reuse

`load_trajectory_split` now accepts optional `expected_manifest_sha256`. When
provided, it must be 64 lowercase hexadecimal characters and match the exact
manifest bytes admitted by that load before an archive is opened. The training
CLI and external benchmark pass the digest returned by their complete verifier
into subsequent split loads. This prevents replacing the manifest between those
operations and silently composing separately verified snapshots.

```python
from ripii.world.external_data import load_trajectory_split, verify_trajectory_dataset

dataset_root = "path/to/dataset-directory"
verification = verify_trajectory_dataset(dataset_root)
train_data, train_record = load_trajectory_split(
    dataset_root,
    "train",
    expected_manifest_sha256=verification["manifest_sha256"],
)
```

The dataset directory contains `manifest.json` and its declared archives. A
single-split load still does not certify the contents of other archives. This
digest binding is not a general protected-evaluation authorization system;
scientific access remains governed by the canonical study contract.

## Retained execution, including the failed demo

The prospective contract is
`development/source_group_v2_20261010/contract.json`. It includes the pre-test
clarification for strict identities and verification-to-reuse binding. The
exact incoming canonical state is retained as `parent_state.json`.

The selected test gate passed **57 cases**, with warnings treated as errors:
37 new cases plus 20 inherited external-data contracts. The new tests cover
within-split windows, every cross-split overlap pair, an extra OOD split,
malformed/missing/duplicate metadata, row-order binding, v1 tensor equality,
retained copied-content rejection, duplicate JSON keys, supplied digest
validation and actual CLI/benchmark manifest-replacement boundaries.

The one initial generated demo completed its single optimizer update and saved
all checkpoints, then failed while reporting the result: the demo incorrectly
treated the checkpoint metadata dictionary returned by `train()` as a path.
`demo_command.json`, both raw logs, `failed_demo_driver.py.txt`, `demo/report.json`
and every dataset/checkpoint remain preserved. That attempt is **FAILED**; its
one-command demo budget is exhausted.

Before any corrective execution, `recovery_contract.json` declared one separate
recovery-only command with zero optimizer updates. The corrected driver loaded
the retained one-step checkpoint, verified its dataset identities and saved raw
evaluation predictions into a fresh directory. `recovery_command.json` and
`recovery/report.json` record completion, zero additional updates, the retained
checkpoint's one completed step, and unchanged hashes for every original demo
artifact. The final fresh-training path was source-reviewed but not rerun.

The fixture retains two overlapping training windows from one actual generated
source episode, a third window from another episode, the complete generated
episodes, all NPZs, the exact manifest and verification, checkpoints, history,
raw predictions and evaluation records. Its declared group counts are two for
training and three each for validation and its generated test split. None is a
protected project outcome, a benchmark claim or an independent scientific
replication. All outputs are artificial engineering evidence.

From the repository root in the existing development environment:

```sh
python -m pytest -q -W error -o addopts= tests/test_source_groups_v2.py tests/test_external_trajectory_data.py tests/test_external_content_overlap.py tests/test_external_container_identity.py
python scripts/demo_source_groups_v2.py --recover-from research/development/source_group_v2_20261010/demo --output /tmp/new-source-group-recovery
```

The output must be fresh. Running the driver without `--recover-from` creates a
new generated dataset and performs one training update; it is a future run and
requires a new recorded budget. This session closes in `closure.json`, retaining
its original failure and separate recovery contract rather than replacing them.

## Project status

This revision follows PR #24 at `4a4ef9dfaf201b05894976195bc28a59b4f7d30b`.
Model architecture, training objective, historical datasets and frozen studies
retain their prior identities. Independent source review checked group admission,
row/hash binding, caller integration and v1 compatibility. The existing negative
and no-advance research dispositions remain closed. No external or confirmatory
campaign was run. The draft uses `[skip ci]` to keep inherited training, broad
performance and artifact campaigns outside this bounded local session.
