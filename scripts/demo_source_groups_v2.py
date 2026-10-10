"""One generated grouped-dataset update with retained inputs and checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ripii.world.experiment import Experiment, evaluate, load_model, train
from ripii.world.external_data import (
    GROUPED_FORMAT,
    load_trajectory_split,
    verify_trajectory_dataset,
)
from ripii.world.physics import make_dataset


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(root):
    root.mkdir()
    artifacts, groups = {}, {}
    for split, id_offset in [("train", 901000), ("validation", 902000), ("test", 903000)]:
        if split == "train":
            source = make_dataset(split, 2, 4, 551)
            data = {
                "states": torch.stack([source["states"][0, :3], source["states"][0, 1:4], source["states"][1, :3]]),
                "actions": torch.stack([source["actions"][0, :2], source["actions"][0, 1:3], source["actions"][1, :2]]),
                "mask": source["mask"][[0, 0, 1]].clone(),
            }
            source_episodes = [0, 0, 1]
            np.savez_compressed(root / "train_source_episodes.npz", **{key: value.numpy() for key, value in source.items()})
        else:
            data = make_dataset(split, 3, 2, 551)
            source_episodes = [0, 1, 2]
        data["ids"] = torch.tensor([id_offset + 1, id_offset + 2, id_offset + 3], dtype=torch.int64)
        path = root / f"{split}.npz"
        np.savez_compressed(path, **{key: value.numpy() for key, value in data.items()})
        raw = path.read_bytes()
        artifacts[split] = {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        groups[split] = [{"trajectory_id": int(row_id), "source_id": "generated.demo." + split, "episode_id": str(episode)} for row_id, episode in zip(data["ids"].tolist(), source_episodes, strict=True)]
    write_json(root / "manifest.json", {
        "format": GROUPED_FORMAT, "dataset_id": "generated.source-group-demo", "version": "1",
        "license": "repository-generated fixture; no external dataset", "units": "synthetic arena units",
        "preprocessing": "train rows0,1 are overlapping two-step windows from one retained source episode; row2 is a different episode",
        "split_policy": "all windows of a declared source episode remain in one partition",
        "observation_dt": 0.05, "source_groups": groups, "artifacts": artifacts,
    })


def artifact_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-from", type=Path, help="Evaluate a retained one-update fixture without training or modifying it")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"scope": "generated engineering fixture; no scientific outcome", "optimizer_updates": 0}
    try:
        torch.set_num_threads(1)
        prior_hashes = None
        if args.recover_from is None:
            dataset = args.output / "dataset"
            prepare(dataset)
            verification = verify_trajectory_dataset(dataset)
        else:
            if args.output.resolve().is_relative_to(args.recover_from.resolve()):
                raise ValueError("recovery output must be outside the retained predecessor")
            prior_hashes = artifact_hashes(args.recover_from)
            report["predecessor_artifact_sha256"] = prior_hashes
            dataset = args.recover_from / "dataset"
            retained = json.loads((args.recover_from / "verification.json").read_text())
            verification = verify_trajectory_dataset(dataset)
            if verification != retained:
                raise ValueError("retained dataset verification changed before recovery")
        write_json(args.output / "verification.json", verification)
        pairs = {split: load_trajectory_split(dataset, split, expected_manifest_sha256=verification["manifest_sha256"]) for split in verification["splits"]}
        if args.recover_from is None:
            cfg = Experiment(steps=1, rollout_steps=1, validate_every=1, hidden=8, batch_size=2, train_scenes=3, eval_scenes=3, train_horizon=2, test_horizon=2, max_objects=8, dt=0.05, data_seed=551)
            model, checkpoint = train(cfg, args.output / "run", variant="graph", seed=553, prepared_datasets=(pairs["train"], pairs["validation"]))
            report["optimizer_updates"] = checkpoint["completed_steps"]
            report["checkpoint"] = "run/best.pt"
        else:
            model, checkpoint = load_model(args.recover_from / "run" / "best.pt")
            report["checkpoint"] = str(args.recover_from / "run" / "best.pt")
        if checkpoint["completed_steps"] != 1 or checkpoint["datasets"] != {"train": pairs["train"][1], "validation": pairs["validation"][1]}:
            raise ValueError("checkpoint does not match the retained one-update grouped fixture")
        report["retained_checkpoint_steps"] = checkpoint["completed_steps"]
        test_data = pairs["test"][0]
        captured = []

        def capture_rollout(_module, _args, prediction):
            if len(captured) < test_data["actions"].shape[1]:
                captured.append(prediction.detach().cpu().clone())

        handle = model.register_forward_hook(capture_rollout)
        try:
            metrics = evaluate(model, test_data, horizons=(1, 2))
        finally:
            handle.remove()
        if len(captured) != test_data["actions"].shape[1]:
            raise ValueError("evaluation did not produce the complete raw rollout")
        predicted = torch.stack([test_data["states"][:, 0], *captured], dim=1)
        np.savez_compressed(args.output / "evaluation_raw.npz", predictions=predicted.numpy(), **{key: value.numpy() for key, value in test_data.items()})
        report["evaluation"] = {"dataset": pairs["test"][1], "metrics": metrics, "raw_output": "evaluation_raw.npz"}
        report["groups_per_split"] = {split: pair[1]["source_groups"]["unique_groups"] for split, pair in pairs.items()}
        if prior_hashes is not None:
            if artifact_hashes(args.recover_from) != prior_hashes:
                raise ValueError("predecessor artifacts changed during recovery")
            report["predecessor_preserved"] = True
        report["status"] = "COMPLETED"
    except BaseException as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        root = Path(__file__).resolve().parents[1]
        report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ["ripii/world/external_data.py", "ripii/world/external_benchmark.py", "ripii/world/cli.py", "ripii/world/experiment.py", "ripii/world/models.py", "ripii/world/physics.py", "scripts/demo_source_groups_v2.py"]}
        report["wall_seconds"] = time.perf_counter() - started
        write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "optimizer_updates": report["optimizer_updates"], "groups_per_split": report["groups_per_split"]}))


if __name__ == "__main__":
    main()
