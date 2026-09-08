from __future__ import annotations

import os
import tempfile

_MATPLOTLIB_TEMP_CACHE: tempfile.TemporaryDirectory[str] | None = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if "MPLCONFIGDIR" not in os.environ:
    _MATPLOTLIB_TEMP_CACHE = tempfile.TemporaryDirectory(
        prefix="ripii-matplotlib-"
    )
    os.environ["MPLCONFIGDIR"] = _MATPLOTLIB_TEMP_CACHE.name
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_script(script: str, argv: list[str]) -> None:
    # Each stage owns libraries with process-global state (PyTorch thread pools and
    # Matplotlib backends). Separate processes make the canonical pipeline match
    # standalone execution and avoid order-dependent warnings or behavior.
    subprocess.run(
        [sys.executable, str(ROOT / script), *argv],
        check=True,
        cwd=ROOT,
        env=os.environ.copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/ripii")
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=3)
    args = parser.parse_args()
    run_script(
        "scripts/train.py",
        ["--config", args.config, "--output-dir", args.output_dir, "--mode", args.mode],
    )
    cfg_path = Path(args.output_dir) / "config.yaml"
    checkpoint = Path(args.output_dir) / "final.pt"
    run_script(
        "scripts/evaluate.py",
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(Path(args.output_dir) / "eval.json"),
        ],
    )
    run_script(
        "scripts/diagnostics.py",
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(Path(args.output_dir) / "diagnostics.png"),
        ],
    )
    if args.run_benchmark:
        benchmark = Path(args.output_dir) / "benchmark.json"
        run_script(
            "scripts/benchmark.py",
            [
                "--config",
                args.config,
                "--seeds",
                "3",
                "--steps",
                str(args.benchmark_steps),
                "--modes",
                "base",
                "--output",
                str(benchmark),
            ],
        )
        run_script(
            "scripts/report.py",
            [
                "--benchmark",
                str(benchmark),
                "--output",
                str(Path(args.output_dir) / "benchmark.md"),
            ],
        )


if __name__ == "__main__":
    main()
