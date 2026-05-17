
from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    subprocess.run([sys.executable, *cmd], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/ripii_suite")
    parser.add_argument("--benchmark-output", type=str, default="runs/ripii_suite/benchmark.json")
    args = parser.parse_args()
    run(["scripts/run_pipeline.py", "--config", args.config, "--output-dir", args.output_dir])
    run(["scripts/benchmark.py", "--config", args.config, "--seeds", "3", "7", "--steps", "3", "--modes", "base", "no_renorm", "no_graph", "no_vq", "no_action", "no_geom", "no_scale", "no_identity", "no_spectral", "no_depth", "no_equiv", "no_moment", "--output", args.benchmark_output])
    run(["scripts/report.py", "--benchmark", args.benchmark_output])


if __name__ == "__main__":
    main()
