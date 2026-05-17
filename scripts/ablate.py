
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ripii.utils.ablation import apply_mode
from ripii.utils.config import load_config, runtime_profile, save_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--mode", type=str, default="no_graph")
    parser.add_argument("--output", type=str, default="runs/ripii/ablation.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cfg = runtime_profile(apply_mode(load_config(args.config), args.mode))
        cfg.output_dir = str(tmpdir / args.mode)
        cfg.amp = False
        cfg.compile_model = False
        cfg.log_every = max(1, cfg.steps)
        cfg.eval_every = max(1, cfg.steps)
        cfg_path = tmpdir / "cfg.yaml"
        save_config(cfg, cfg_path)
        subprocess.run([sys.executable, "scripts/train.py", "--config", str(cfg_path), "--mode", "base"], check=True, cwd=ROOT)
        checkpoint = Path(cfg.output_dir) / "final.pt"
        proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--config", str(cfg_path), "--checkpoint", str(checkpoint)], check=True, capture_output=True, text=True, cwd=ROOT)
        summary = json.loads(proc.stdout.strip())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "summary": summary}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
