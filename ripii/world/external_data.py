"""Manifest-verified interchange format for object-state trajectory datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .data import validate_tensor_dataset

FORMAT = "ripii-trajectory-dataset-v1"
REQUIRED_SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("external dataset manifest is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("external dataset manifest is invalid JSON") from exc
    required_strings = (
        "dataset_id",
        "version",
        "license",
        "units",
        "preprocessing",
        "split_policy",
    )
    if (
        not isinstance(value, dict)
        or value.get("format") != FORMAT
        or any(
            not isinstance(value.get(key), str) or not value[key].strip()
            for key in required_strings
        )
        or not isinstance(value.get("artifacts"), dict)
        or not isinstance(value.get("observation_dt"), (int, float))
        or isinstance(value.get("observation_dt"), bool)
        or not np.isfinite(value["observation_dt"])
        or value["observation_dt"] <= 0
        or not set(REQUIRED_SPLITS) <= set(value["artifacts"])
    ):
        raise ValueError("external dataset manifest violates the required schema")
    return value


def _artifact_path(root: Path, entry: dict, split: str) -> Path:
    relative = entry.get("path") if isinstance(entry, dict) else None
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    path = root / relative if isinstance(relative, str) else root
    if (
        pure is None
        or pure.is_absolute()
        or ".." in pure.parts
        or path.is_symlink()
        or not path.is_file()
        or not isinstance(entry.get("bytes"), int)
        or isinstance(entry.get("bytes"), bool)
        or entry["bytes"] < 1
        or path.stat().st_size != entry["bytes"]
        or not isinstance(entry.get("sha256"), str)
        or _sha256(path) != entry["sha256"]
    ):
        raise ValueError(f"external dataset artifact failed verification: {split}")
    return path


def load_trajectory_split(
    root: str | Path, split: str
) -> tuple[dict[str, torch.Tensor], dict]:
    root = Path(root).resolve()
    manifest = _manifest(root)
    if split not in manifest["artifacts"]:
        raise ValueError(f"external dataset does not declare split: {split}")
    entry = manifest["artifacts"][split]
    path = _artifact_path(root, entry, split)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"states", "actions", "mask", "ids"}:
                raise ValueError("trajectory NPZ contains unexpected arrays")
            data = {
                "states": torch.from_numpy(archive["states"].copy()),
                "actions": torch.from_numpy(archive["actions"].copy()),
                "mask": torch.from_numpy(archive["mask"].copy()),
                "ids": torch.from_numpy(archive["ids"].copy()),
            }
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"external trajectory split is invalid: {split}") from exc
    states, actions = data["states"], data["actions"]
    if states.ndim != 4 or actions.ndim != 4:
        raise ValueError("external trajectory arrays have invalid ranks")
    scenes, observations, max_objects, features = states.shape
    if features != 6 or actions.shape != (scenes, observations - 1, max_objects, 2):
        raise ValueError("external trajectory state/action dimensions do not align")
    validate_tensor_dataset(scenes, observations - 1, max_objects, data)
    record = {
        "format": FORMAT,
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "license": manifest["license"],
        "units": manifest["units"],
        "preprocessing": manifest["preprocessing"],
        "split_policy": manifest["split_policy"],
        "observation_dt": float(manifest["observation_dt"]),
        "split": split,
        "scenes": scenes,
        "horizon": observations - 1,
        "max_objects": max_objects,
        "source_path": entry["path"],
        "source_sha256": entry["sha256"],
        "manifest_sha256": _sha256(root / "manifest.json"),
    }
    return data, record


def verify_trajectory_dataset(root: str | Path) -> dict:
    root = Path(root).resolve()
    manifest = _manifest(root)
    records = {}
    ids_by_split = {}
    for split in manifest["artifacts"]:
        data, record = load_trajectory_split(root, split)
        records[split] = record
        ids_by_split[split] = set(data["ids"].tolist())
    for index, left in enumerate(ids_by_split):
        for right in list(ids_by_split)[index + 1 :]:
            if ids_by_split[left] & ids_by_split[right]:
                raise ValueError(f"trajectory IDs overlap between {left} and {right}")
    return {
        "status": "PASS",
        "format": FORMAT,
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "manifest_sha256": _sha256(root / "manifest.json"),
        "splits": records,
    }
