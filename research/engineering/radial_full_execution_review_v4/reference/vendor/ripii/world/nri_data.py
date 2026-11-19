"""Fail-closed adapter for pinned NRI spring and charged-particle trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import validate_tensor_dataset

DOMAINS = ("springs", "charged")
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(root: Path) -> tuple[dict, str]:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("NRI manifest is missing or unsafe")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("NRI manifest is unreadable or invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "ripii-nri-data-v1"
        or manifest.get("domains") != list(DOMAINS)
        or not isinstance(manifest.get("generation"), dict)
        or not isinstance(manifest.get("upstream"), dict)
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ValueError("invalid NRI manifest schema")
    return manifest, _sha256(path)


def load_nri_dataset(
    root: Path,
    domain: str,
    split: str,
    scenes: int,
    horizon: int,
    max_objects: int = 8,
) -> tuple[dict[str, torch.Tensor], dict]:
    if (
        domain not in DOMAINS
        or split not in SPLITS
        or not isinstance(scenes, int)
        or isinstance(scenes, bool)
        or not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or not isinstance(max_objects, int)
        or isinstance(max_objects, bool)
        or scenes < 1
        or horizon < 1
        or not 5 <= max_objects <= 16
    ):
        raise ValueError("invalid NRI dataset request")
    root = Path(root).resolve()
    manifest, manifest_sha256 = _read_manifest(root)
    name = f"{domain}_{split}.npz"
    entries = [entry for entry in manifest["artifacts"] if entry.get("path") == name]
    if len(entries) != 1:
        raise ValueError(f"NRI manifest does not declare exactly one {name}")
    path = root / name
    entry = entries[0]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != entry.get("bytes")
        or _sha256(path) != entry.get("sha256")
    ):
        raise ValueError(f"NRI data hash or size mismatch: {name}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"loc", "vel", "edges", "seed"}:
                raise ValueError(
                    "NRI archive schema differs from the manifest contract"
                )
            loc = np.asarray(archive["loc"], dtype=np.float32)
            vel = np.asarray(archive["vel"], dtype=np.float32)
            edges = np.asarray(archive["edges"], dtype=np.float32)
            split_seed = int(archive["seed"])
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"NRI archive is unreadable or invalid: {name}") from exc
    generation = manifest["generation"]
    available_scenes = (
        generation["train_scenes"] if split == "train" else generation["eval_scenes"]
    )
    available_horizon = (
        generation["test_horizon"] if split == "test" else generation["train_horizon"]
    )
    expected = (available_scenes, available_horizon + 1, 2, 5)
    expected_seed = (
        int(generation["base_seed"])
        + int(generation["domain_seed_offsets"][domain])
        + int(generation["split_seed_offsets"][split])
    )
    if (
        loc.shape != expected
        or vel.shape != expected
        or edges.shape != (available_scenes, 5, 5)
        or scenes > available_scenes
        or horizon > available_horizon
        or split_seed != expected_seed
        or not np.isfinite(loc).all()
        or not np.isfinite(vel).all()
        or not np.isfinite(edges).all()
    ):
        raise ValueError("NRI tensors violate the declared dimensions or finiteness")
    scale = float(generation["box_size"])
    if scale != 5.0 or float(generation["observation_dt"]) != 0.05:
        raise ValueError(
            "NRI physical scale or time step differs from the model contract"
        )
    state = torch.zeros(scenes, horizon + 1, max_objects, 6, dtype=torch.float32)
    state[:, :, :5, :2] = (
        torch.from_numpy(loc[:scenes, : horizon + 1].transpose(0, 1, 3, 2).copy())
        / scale
    )
    state[:, :, :5, 2:4] = (
        torch.from_numpy(vel[:scenes, : horizon + 1].transpose(0, 1, 3, 2).copy())
        / scale
    )
    state[:, :, :5, 4] = 0.04
    state[:, :, :5, 5] = 1.0
    mask = torch.zeros(scenes, max_objects, dtype=torch.bool)
    mask[:, :5] = True
    data = {
        "states": state,
        "actions": torch.zeros(scenes, horizon, max_objects, 2),
        "mask": mask,
        "ids": torch.arange(scenes, dtype=torch.int64) + split_seed * 1_000_000,
    }
    validate_tensor_dataset(scenes, horizon, max_objects, data)
    record = {
        "format": "ripii-external-dataset-record-v1",
        "dataset_id": f"nri.{domain}",
        "version": manifest["upstream"]["commit"],
        "license": manifest["upstream"]["license"],
        "split": split,
        "scenes": scenes,
        "horizon": horizon,
        "max_objects": max_objects,
        "source_archive": name,
        "source_sha256": entry["sha256"],
        "manifest_sha256": manifest_sha256,
        "simulator_sha256": manifest["upstream"]["simulator_sha256"],
        "observation_dt": generation["observation_dt"],
        "preprocessing": manifest["preprocessing"],
        "split_policy": manifest["split_policy"],
        "edge_labels_used": False,
        "field_provenance": {
            "position": "observed_from_nri_loc_then_divided_by_box_size",
            "velocity": "observed_from_nri_vel_then_divided_by_box_size",
            "radius": "constructed_constant_0.04_not_observed",
            "mass": "constructed_constant_1.0_not_observed",
            "action": "constructed_zero_not_observed",
            "mask": "constructed_first_five_slots_live",
            "edge_labels": "retained_in_source_archive_but_not_model_input",
        },
        "metric_applicability": {
            "position": True,
            "velocity": True,
            "property_drift": False,
            "action_response": False,
            "relation_recovery": False,
        },
        "claim_boundary": (
            "external simulator development data; not fixed real-world observations "
            "and not confirmatory external validation"
        ),
    }
    return data, record
