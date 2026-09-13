from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from .experiment import Experiment, benchmark, capture, train, verify, verify_capsule
from .external_benchmark import benchmark_trajectory_dataset
from .external_data import load_trajectory_split, verify_trajectory_dataset
from .inference import (
    WorldPredictor,
    inspect_json,
    load_inference_npz,
    save_prediction_npz,
)
from .models import VARIANTS


def add_experiment(parser):
    defaults = Experiment()
    for field in fields(Experiment):
        default = getattr(defaults, field.name)
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(default), default=default
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train, benchmark, and interact with RIPII object-state world models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "benchmark"):
        sub = subparsers.add_parser(name)
        add_experiment(sub)
        sub.add_argument("--output", type=Path, required=True)
        if name == "train":
            sub.add_argument("--model", choices=VARIANTS, default="graph")
            sub.add_argument(
                "--bottleneck",
                choices=("continuous", "fsq", "vq"),
                default="continuous",
            )
            sub.add_argument("--seed", type=int, default=3)
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--dataset-root", type=Path)
        else:
            sub.add_argument(
                "--models", choices=VARIANTS, nargs="+", default=list(VARIANTS)
            )
            sub.add_argument(
                "--bottlenecks",
                choices=("continuous", "fsq", "vq"),
                nargs="+",
                default=["continuous"],
            )
            sub.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    sub = subparsers.add_parser("benchmark-dataset")
    add_experiment(sub)
    sub.add_argument("--dataset-root", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--models", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    sub.add_argument(
        "--bottlenecks",
        choices=("continuous", "fsq", "vq"),
        nargs="+",
        default=["continuous"],
    )
    sub.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    sub = subparsers.add_parser("demo")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument(
        "--export", type=Path, help="render a PNG without opening a window"
    )
    sub.add_argument("--seed", type=int, default=42)
    sub = subparsers.add_parser("verify")
    sub.add_argument("directory", type=Path)
    sub = subparsers.add_parser("capture")
    sub.add_argument("directory", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    sub = subparsers.add_parser("verify-capsule")
    sub.add_argument("capsule", type=Path)
    sub = subparsers.add_parser("verify-dataset")
    sub.add_argument("directory", type=Path)
    sub = subparsers.add_parser("evaluate")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--dataset-root", type=Path, required=True)
    sub.add_argument("--split", default="test")
    sub.add_argument("--output", type=Path)
    sub = subparsers.add_parser("inspect")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--device", default="cpu")
    for name in ("predict", "rollout"):
        sub = subparsers.add_parser(name)
        sub.add_argument("checkpoint", type=Path)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command in {"train", "benchmark", "benchmark-dataset"}:
        cfg = Experiment(
            **{field.name: getattr(args, field.name) for field in fields(Experiment)}
        )
        if args.command == "train":
            prepared = None
            if args.dataset_root is not None:
                verify_trajectory_dataset(args.dataset_root)
                train_pair = load_trajectory_split(args.dataset_root, "train")
                validation_pair = load_trajectory_split(
                    args.dataset_root, "validation"
                )
                if (
                    train_pair[1]["max_objects"] != validation_pair[1]["max_objects"]
                    or train_pair[1]["observation_dt"]
                    != validation_pair[1]["observation_dt"]
                ):
                    raise ValueError(
                        "external train and validation contracts do not align"
                    )
                cfg.train_scenes = train_pair[1]["scenes"]
                cfg.train_horizon = train_pair[1]["horizon"]
                cfg.eval_scenes = validation_pair[1]["scenes"]
                cfg.max_objects = train_pair[1]["max_objects"]
                cfg.dt = train_pair[1]["observation_dt"]
                prepared = (train_pair, validation_pair)
            _, checkpoint = train(
                cfg,
                args.output,
                args.model,
                args.seed,
                args.bottleneck,
                resume=args.resume,
                prepared_datasets=prepared,
            )
            result = {
                "checkpoint": str(args.output / "best.pt"),
                "selected_step": checkpoint["completed_steps"],
                "validation_score": checkpoint["best_validation"],
            }
        elif args.command == "benchmark":
            report = benchmark(
                cfg, args.output, args.seeds, args.models, args.bottlenecks
            )
            result = {
                "report": str(args.output / "report.md"),
                "decision": report["decision"],
            }
        else:
            report = benchmark_trajectory_dataset(
                args.dataset_root,
                args.output,
                cfg,
                tuple(args.seeds),
                tuple(args.models),
                tuple(args.bottlenecks),
            )
            result = {
                "report": str(args.output / "report.md"),
                "decision": report["decision"],
            }
    elif args.command == "verify":
        result = verify(args.directory)
    elif args.command == "capture":
        payload = capture(args.directory, args.output)
        result = {
            "output": str(args.output),
            "decision": payload["decision"],
            "result_sha256": payload["result_sha256"],
        }
    elif args.command == "verify-capsule":
        result = verify_capsule(args.capsule)
    elif args.command == "verify-dataset":
        result = verify_trajectory_dataset(args.directory)
    elif args.command == "evaluate":
        from .experiment import evaluate, write_json

        predictor = WorldPredictor.from_checkpoint(args.checkpoint)
        data, dataset = load_trajectory_split(args.dataset_root, args.split)
        if (
            predictor.model.max_objects != dataset["max_objects"]
            or predictor.model.dt != dataset["observation_dt"]
        ):
            raise ValueError("checkpoint and dataset object/time contracts differ")
        metrics = evaluate(predictor.model, data)
        result = {"dataset": dataset, "metrics": metrics}
        if args.output is not None:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(f"refusing existing output: {args.output}")
            write_json(args.output, result)
    elif args.command == "demo":
        from .demo import main as run_demo

        run_demo(args.checkpoint, args.export, args.seed)
        result = {
            "checkpoint": str(args.checkpoint),
            "export": str(args.export) if args.export else None,
        }
    elif args.command == "inspect":
        predictor = WorldPredictor.from_checkpoint(args.checkpoint, args.device)
        print(inspect_json(predictor))
        return
    else:
        predictor = WorldPredictor.from_checkpoint(args.checkpoint, args.device)
        payload = load_inference_npz(args.input)
        if args.command == "predict":
            if "action" not in payload:
                raise ValueError("predict requires an action array")
            states = predictor.predict(
                payload["state"], payload["action"], payload["mask"]
            )
        else:
            if "actions" not in payload:
                raise ValueError("rollout requires an actions array")
            states = predictor.rollout(
                payload["state"], payload["actions"], payload["mask"]
            )
        save_prediction_npz(args.output, states)
        result = {
            "checkpoint": str(args.checkpoint),
            "input": str(args.input),
            "output": str(args.output),
            "shape": list(states.shape),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
