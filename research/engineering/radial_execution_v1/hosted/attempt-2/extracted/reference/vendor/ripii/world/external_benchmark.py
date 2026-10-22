"""Generic, manifested development benchmark for trajectory datasets."""

from __future__ import annotations

import hashlib
import platform
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .experiment import (
    ANALYTIC_BASELINES,
    CAPACITY_MATCH_TOLERANCE,
    Experiment,
    evaluate,
    train,
    widths,
    write_json,
)
from .external_data import load_trajectory_split, verify_trajectory_dataset
from .models import VARIANTS

FORMAT = "ripii-external-trajectory-benchmark-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(output: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "ripii").rglob("*.py"))
    paths.extend(path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file())
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        hashes[str(relative)] = _sha256(path)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return hashes


def benchmark_trajectory_dataset(
    root: str | Path,
    output: str | Path,
    cfg: Experiment,
    seeds: tuple[int, ...] = (3, 7, 11),
    variants: tuple[str, ...] = VARIANTS,
    bottlenecks: tuple[str, ...] = ("continuous",),
) -> dict:
    root, output = Path(root).resolve(), Path(output)
    verification = verify_trajectory_dataset(root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing benchmark output: {output}")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        or not variants
        or len(set(variants)) != len(variants)
        or any(variant not in VARIANTS for variant in variants)
        or not bottlenecks
        or len(set(bottlenecks)) != len(bottlenecks)
        or any(item not in {"continuous", "fsq", "vq"} for item in bottlenecks)
    ):
        raise ValueError("benchmark grid must contain valid unique values")
    loaded = {
        split: load_trajectory_split(root, split)
        for split in verification["splits"]
    }
    train_pair, validation_pair, test_pair = (
        loaded["train"],
        loaded["validation"],
        loaded["test"],
    )
    contracts = {
        (record["max_objects"], record["observation_dt"])
        for _, record in loaded.values()
    }
    if len(contracts) != 1:
        raise ValueError("all dataset splits must share object and timestep contracts")
    cfg = replace(
        cfg,
        train_scenes=train_pair[1]["scenes"],
        train_horizon=train_pair[1]["horizon"],
        eval_scenes=validation_pair[1]["scenes"],
        test_horizon=test_pair[1]["horizon"],
        max_objects=train_pair[1]["max_objects"],
        dt=train_pair[1]["observation_dt"],
    )
    cfg.validate()
    if cfg.rollout_steps > validation_pair[1]["horizon"]:
        raise ValueError("rollout_steps exceeds the external validation horizon")
    capacity = {item: widths(cfg, variants, item) for item in bottlenecks}
    mismatched = {
        f"{variant}_{bottleneck}": details["relative_error"]
        for bottleneck, by_variant in capacity.items()
        for variant, details in by_variant.items()
        if details["relative_error"] > CAPACITY_MATCH_TOLERANCE
    }
    if mismatched:
        raise ValueError(f"capacity matching exceeds 5% tolerance: {mismatched}")
    output.mkdir(parents=True)
    source_hashes = _source_snapshot(output)
    protocol = {
        "format": FORMAT,
        "status": "development_benchmark",
        "dataset_verification": verification,
        "experiment": asdict(cfg),
        "seeds": list(seeds),
        "variants": list(variants),
        "bottlenecks": list(bottlenecks),
        "capacity": capacity,
        "source_sha256": source_hashes,
        "selection": "minimum validation position RMSE + 0.25 * velocity RMSE",
        "claim_boundary": "development evidence; no positive or population claim",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    write_json(output / "protocol.json", protocol)
    rows = []
    evaluation_splits = {
        split: pair[0]
        for split, pair in loaded.items()
        if split not in {"train", "validation"}
    }
    for bottleneck in bottlenecks:
        for variant in variants:
            for seed in seeds:
                directory = output / f"{variant}_{bottleneck}" / f"seed_{seed}"
                model, checkpoint = train(
                    cfg,
                    directory,
                    variant,
                    seed,
                    bottleneck,
                    capacity[bottleneck][variant]["hidden"],
                    prepared_datasets=(train_pair, validation_pair),
                )
                metrics = {
                    split: evaluate(model, data)
                    for split, data in evaluation_splits.items()
                }
                row = {
                    "variant": variant,
                    "bottleneck": bottleneck,
                    "seed": seed,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "selected_step": checkpoint["completed_steps"],
                    "train_seconds": checkpoint["train_seconds"],
                    "metrics": metrics,
                }
                write_json(directory / "evaluation.json", row)
                rows.append(row)
    analytic = {}
    for name, factory in ANALYTIC_BASELINES.items():
        baseline = factory(cfg.dt) if name == "force_kinematic" else factory()
        analytic[name] = {
            split: evaluate(baseline, data)
            for split, data in evaluation_splits.items()
        }
    summary = {
        "format": FORMAT,
        "protocol_sha256": _sha256(output / "protocol.json"),
        "runs": rows,
        "analytic_baselines": analytic,
        "decision": "no_claim",
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output / "summary.json", summary)
    report = [
        "# External trajectory development benchmark",
        "",
        "Decision: **no_claim**. This generic run has no preregistered positive gate.",
        "",
        "| Model | Bottleneck | Seed | Test position RMSE | Parameters |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['variant']} | {row['bottleneck']} | {row['seed']} | "
            f"{row['metrics']['test']['position_rmse']:.6f} | {row['parameters']} |"
        )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_paths = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "format": FORMAT,
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(output / "manifest.json", manifest)
    return summary
