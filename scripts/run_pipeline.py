
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_script(script: str, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = [script, *argv]
    try:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/ripii")
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=3)
    args = parser.parse_args()
    run_script("scripts/train.py", ["--config", args.config, "--output-dir", args.output_dir, "--mode", args.mode])
    cfg_path = Path(args.config)
    checkpoint = Path(args.output_dir) / "final.pt"
    run_script("scripts/evaluate.py", ["--config", str(cfg_path), "--checkpoint", str(checkpoint)])
    run_script("scripts/diagnostics.py", ["--config", str(cfg_path), "--checkpoint", str(checkpoint), "--output", str(Path(args.output_dir) / "diagnostics.png")])
    if args.run_benchmark:
        benchmark = Path(args.output_dir) / "benchmark.json"
        run_script("scripts/benchmark.py", ["--config", args.config, "--seeds", "3", "--steps", str(args.benchmark_steps), "--modes", "base", "--output", str(benchmark)])
        run_script("scripts/report.py", ["--benchmark", str(benchmark), "--output", str(Path(args.output_dir) / "benchmark.md")])


if __name__ == "__main__":
    main()
