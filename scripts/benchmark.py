
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ripii.utils.ablation import apply_mode
from ripii.utils.config import load_config, runtime_profile, save_config
from ripii.utils.reporting import summarize_benchmark, render_markdown


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *cmd], check=True, capture_output=True, text=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    parser.add_argument("--modes", type=str, nargs="+", default=["base", "no_renorm", "no_graph", "no_vq", "no_action", "no_geom", "no_scale", "no_identity", "no_spectral", "no_depth", "no_equiv", "no_moment"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=str, default="runs/ripii/benchmark.json")
    args = parser.parse_args()
    base = load_config(args.config)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for mode in args.modes:
            for seed in args.seeds:
                cfg = runtime_profile(apply_mode(base, mode))
                cfg.seed = int(seed)
                cfg.steps = 1 if (cfg.input_dim <= 32 and args.steps <= 3) else int(args.steps)
                cfg.output_dir = str(tmpdir / f"{mode}_{seed}")
                cfg.compile_model = False
                cfg.amp = False
                cfg.log_every = max(1, cfg.steps)
                cfg.eval_every = max(1, cfg.steps)
                cfg_path = tmpdir / f"cfg_{mode}_{seed}.yaml"
                save_config(cfg, cfg_path)
                run(["scripts/train.py", "--config", str(cfg_path), "--mode", "base"])
                checkpoint = Path(cfg.output_dir) / "final.pt"
                eval_proc = run(["scripts/evaluate.py", "--config", str(cfg_path), "--checkpoint", str(checkpoint)])
                eval_summary = json.loads(eval_proc.stdout.strip())
                results.append({"mode": mode, "seed": seed, **eval_summary})
    numeric_keys = sorted({k for row in results for k, v in row.items() if isinstance(v, (int, float)) and k not in {"seed"}})
    summary: dict[str, Any] = {"runs": results}
    grouped = {}
    for row in results:
        grouped.setdefault(row["mode"], []).append(row)
    summary["by_mode"] = {}
    for mode, rows in grouped.items():
        mode_summary = {}
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if key in r]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                mode_summary[f"{key}_mean"] = mean
                mode_summary[f"{key}_std"] = std
        summary["by_mode"][mode] = mode_summary
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in results for k in row.keys()}))
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    report_path = out.with_suffix(".md")
    report_path.write_text(render_markdown(summarize_benchmark(summary)), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
