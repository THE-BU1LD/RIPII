from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ripii.world.experiment import Experiment
from ripii.world.external_benchmark import benchmark_trajectory_dataset
from ripii.world.external_data import (
    FORMAT,
    load_trajectory_split,
    verify_trajectory_dataset,
)
from ripii.world.physics import make_dataset


def build_dataset(root: Path) -> Path:
    root.mkdir()
    artifacts = {}
    for split in ("train", "validation", "test"):
        data = make_dataset(split, 3, 2, 17)
        path = root / f"{split}.npz"
        np.savez_compressed(path, **{key: value.numpy() for key, value in data.items()})
        artifacts[split] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "dataset_id": "example.trajectories",
                "version": "1",
                "license": "CC-BY-4.0",
                "units": "position=m, velocity=m/s, radius=m, mass=kg, force=N",
                "preprocessing": "none",
                "split_policy": "disjoint trajectory IDs",
                "observation_dt": 0.05,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_external_trajectory_dataset_verifies_and_loads(tmp_path: Path):
    root = build_dataset(tmp_path / "dataset")
    verification = verify_trajectory_dataset(root)
    data, record = load_trajectory_split(root, "test")
    assert verification["status"] == "PASS"
    assert record["license"] == "CC-BY-4.0"
    assert data["states"].shape == (3, 3, 8, 6)


def test_external_trajectory_dataset_detects_tampering(tmp_path: Path):
    root = build_dataset(tmp_path / "dataset")
    with (root / "test.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="failed verification"):
        verify_trajectory_dataset(root)


def test_external_dataset_cli_trains_and_evaluates(tmp_path: Path):
    root = build_dataset(tmp_path / "dataset")
    output = tmp_path / "run"
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ripii.world",
            "train",
            "--dataset-root",
            str(root),
            "--output",
            str(output),
            "--model",
            "graph",
            "--steps",
            "1",
            "--rollout-steps",
            "1",
            "--validate-every",
            "1",
            "--hidden",
            "8",
            "--batch-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    evaluation = tmp_path / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ripii.world",
            "evaluate",
            str(output / "best.pt"),
            "--dataset-root",
            str(root),
            "--output",
            str(evaluation),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(evaluation.read_text(encoding="utf-8"))
    assert result["dataset"]["dataset_id"] == "example.trajectories"
    assert result["metrics"]["position_rmse"] >= 0


def test_external_dataset_benchmark_records_raw_runs(tmp_path: Path):
    root = build_dataset(tmp_path / "dataset")
    output = tmp_path / "benchmark"
    summary = benchmark_trajectory_dataset(
        root,
        output,
        Experiment(steps=1, rollout_steps=1, validate_every=1, hidden=64),
        seeds=(3,),
        variants=("graph",),
        bottlenecks=("continuous",),
    )
    assert summary["decision"] == "no_claim"
    assert len(summary["runs"]) == 1
    assert (output / "protocol.json").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "manifest.json").is_file()
