from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ripii.world.experiment import Experiment, train
from ripii.world.nri_data import load_nri_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> None:
    artifacts = []
    for domain_index, domain in enumerate(("springs", "charged")):
        for split_index, split in enumerate(("train", "validation", "test")):
            scenes, horizon = (
                (4, 4) if split == "train" else (2, 4 if split == "validation" else 6)
            )
            loc = np.zeros((scenes, horizon + 1, 2, 5), dtype=np.float32)
            vel = np.zeros_like(loc)
            loc[:, :, 0, :] = domain_index + np.arange(horizon + 1)[None, :, None]
            vel[:, :, 1, :] = split_index + 1
            edges = np.zeros((scenes, 5, 5), dtype=np.float32)
            path = root / f"{domain}_{split}.npz"
            np.savez_compressed(
                path,
                loc=loc,
                vel=vel,
                edges=edges,
                seed=np.asarray(100 + 10 * domain_index + split_index, dtype=np.int64),
            )
            artifacts.append(
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "format": "ripii-nri-data-v1",
        "domains": ["springs", "charged"],
        "upstream": {
            "commit": "pinned-test-commit",
            "license": "MIT",
            "simulator_sha256": "a" * 64,
        },
        "generation": {
            "base_seed": 100,
            "domain_seed_offsets": {"springs": 0, "charged": 10},
            "split_seed_offsets": {"train": 0, "validation": 1, "test": 2},
            "train_scenes": 4,
            "eval_scenes": 2,
            "train_horizon": 4,
            "test_horizon": 6,
            "box_size": 5.0,
            "observation_dt": 0.05,
        },
        "preprocessing": "fixed scale",
        "split_policy": "disjoint RNG domains",
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_nri_adapter_maps_state_without_using_edges(tmp_path: Path) -> None:
    _fixture(tmp_path)
    data, record = load_nri_dataset(tmp_path, "springs", "test", 2, 6)
    assert data["states"].shape == (2, 7, 8, 6)
    assert data["actions"].shape == (2, 6, 8, 2)
    assert data["mask"].sum(1).tolist() == [5, 5]
    assert torch.allclose(data["states"][:, 1, :5, 0], torch.full((2, 5), 0.2))
    assert torch.allclose(data["states"][:, :, :5, 3], torch.full((2, 7, 5), 0.6))
    assert (data["states"][:, :, 5:] == 0).all()
    assert (data["actions"] == 0).all()
    assert record["edge_labels_used"] is False
    assert record["field_provenance"]["radius"].endswith("not_observed")
    assert record["field_provenance"]["edge_labels"].endswith("not_model_input")
    assert record["metric_applicability"]["property_drift"] is False
    assert record["metric_applicability"]["relation_recovery"] is False
    assert record["license"] == "MIT"
    assert record["claim_boundary"].startswith("external simulator development")


def test_nri_adapter_fails_on_corruption_and_contract_drift(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "charged_test.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        load_nri_dataset(tmp_path, "charged", "test", 2, 6)
    _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"]["observation_dt"] = 0.1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="time step"):
        load_nri_dataset(tmp_path, "springs", "train", 4, 4)


def test_training_uses_and_records_prepared_external_data(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _fixture(data_root)
    train_pair = load_nri_dataset(data_root, "springs", "train", 4, 4)
    validation_pair = load_nri_dataset(data_root, "springs", "validation", 2, 2)
    cfg = Experiment(
        steps=2,
        train_scenes=4,
        eval_scenes=2,
        train_horizon=4,
        test_horizon=4,
        rollout_steps=2,
        batch_size=2,
        hidden=16,
        validate_every=1,
    )
    _, checkpoint = train(
        cfg,
        tmp_path / "run",
        variant="graph",
        seed=3,
        prepared_datasets=(train_pair, validation_pair),
    )
    assert checkpoint["datasets"]["train"]["dataset_id"] == "nri.springs"
    config = json.loads((tmp_path / "run/config.json").read_text(encoding="utf-8"))
    assert config["datasets"] == checkpoint["datasets"]
