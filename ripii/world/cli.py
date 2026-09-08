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
            sub.add_argument("--model", choices=VARIANTS, default="multiscale")
            sub.add_argument(
                "--bottleneck",
                choices=("continuous", "fsq", "vq"),
                default="continuous",
            )
            sub.add_argument("--seed", type=int, default=3)
            sub.add_argument("--resume", type=Path)
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
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command in {"train", "benchmark"}:
        cfg = Experiment(
            **{field.name: getattr(args, field.name) for field in fields(Experiment)}
        )
        if args.command == "train":
            _, checkpoint = train(
                cfg,
                args.output,
                args.model,
                args.seed,
                args.bottleneck,
                resume=args.resume,
            )
            result = {
                "checkpoint": str(args.output / "best.pt"),
                "selected_step": checkpoint["completed_steps"],
                "validation_score": checkpoint["best_validation"],
            }
        else:
            report = benchmark(
                cfg, args.output, args.seeds, args.models, args.bottlenecks
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
    else:
        from .demo import main as run_demo

        run_demo(args.checkpoint, args.export, args.seed)
        result = {
            "checkpoint": str(args.checkpoint),
            "export": str(args.export) if args.export else None,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
