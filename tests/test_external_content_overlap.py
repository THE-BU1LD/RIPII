from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ripii.world.external_data import (
    FORMAT,
    TRAJECTORY_FINGERPRINT,
    verify_trajectory_dataset,
)
from ripii.world.physics import make_dataset


def _write_split(root: Path, split: str, arrays: dict[str, np.ndarray]) -> None:
    path = root / f"{split}.npz"
    np.savez_compressed(path, **arrays)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][split] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _read_split(root: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "dataset_id": "unit-test.generated-content",
                "version": "1",
                "license": "NOASSERTION",
                "units": "synthetic units",
                "preprocessing": "none",
                "split_policy": "disjoint trajectories",
                "observation_dt": 0.05,
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        arrays = make_dataset(split, 3, 2, 17)
        _write_split(root, split, {key: value.numpy() for key, value in arrays.items()})
    return root


@pytest.mark.parametrize("destination", ["validation", "test", "extra_ood"])
def test_copied_training_trajectories_cannot_be_renumbered_into_a_split(
    dataset: Path, destination: str
) -> None:
    copied = _read_split(dataset, "train")
    copied["ids"] += 999_000_000
    _write_split(dataset, destination, copied)
    with pytest.raises(ValueError, match=f"contents overlap between train and {destination}"):
        verify_trajectory_dataset(dataset)


def test_one_duplicate_among_unrelated_rows_is_detected(dataset: Path) -> None:
    train = _read_split(dataset, "train")
    test = _read_split(dataset, "test")
    for key in ("states", "actions", "mask"):
        test[key][1] = train[key][2]
    _write_split(dataset, "test", test)
    with pytest.raises(ValueError, match="contents overlap"):
        verify_trajectory_dataset(dataset)


def test_storage_dtype_signed_zero_and_padding_do_not_hide_duplicates(
    dataset: Path,
) -> None:
    copied = _read_split(dataset, "train")
    copied["ids"] += 999_000_000
    for key in ("states", "actions"):
        values = copied[key].astype(np.float64)
        values[values == 0] = -0.0
        # Add two all-zero padded object slots at the front. Live-object order
        # and physical observations are identical to the original trajectory.
        copied[key] = np.pad(values, ((0, 0), (0, 0), (2, 0), (0, 0)))
    copied["mask"] = np.pad(copied["mask"], ((0, 0), (2, 0)))
    _write_split(dataset, "test", copied)
    with pytest.raises(ValueError, match="contents overlap"):
        verify_trajectory_dataset(dataset)


@pytest.mark.parametrize("changed_field", ["states", "actions"])
def test_distinct_live_values_do_not_trigger_exact_overlap(
    dataset: Path, changed_field: str
) -> None:
    copied = _read_split(dataset, "train")
    copied["ids"] += 999_000_000
    for scene in range(copied["ids"].size):
        live = np.flatnonzero(copied["mask"][scene])[0]
        copied[changed_field][scene, 0, live, 0] += 0.125
    _write_split(dataset, "test", copied)
    result = verify_trajectory_dataset(dataset)
    assert result["status"] == "PASS"
    assert result["content_overlap_check"]["fingerprint_schema"] == TRAJECTORY_FINGERPRINT


def test_fingerprints_are_reproducible_and_row_order_independent(dataset: Path) -> None:
    original = verify_trajectory_dataset(dataset)
    copied = _read_split(dataset, "test")
    _write_split(dataset, "test", {key: value[::-1] for key, value in copied.items()})
    reordered = verify_trajectory_dataset(dataset)
    left = original["splits"]["test"]["content_verification"]
    right = reordered["splits"]["test"]["content_verification"]
    assert left == right
    assert left["unique_trajectories"] == 3
    assert len(left["cohort_sha256"]) == 64
    assert original["manifest_sha256"] != reordered["manifest_sha256"]


def test_existing_id_overlap_rule_remains_active(dataset: Path) -> None:
    train = _read_split(dataset, "train")
    test = _read_split(dataset, "test")
    test["ids"][0] = train["ids"][0]
    _write_split(dataset, "test", test)
    with pytest.raises(ValueError, match="IDs overlap"):
        verify_trajectory_dataset(dataset)
