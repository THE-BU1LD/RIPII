from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ripii.models.baselines import match_plain_autoencoder_capacity
from ripii.models.factory import build_model
from ripii.utils.ablation import apply_mode
from ripii.utils.config import load_config, runtime_profile, save_config
from ripii.utils.reporting import render_markdown, summarize_benchmark
from ripii.utils.seed import seed_everything


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *cmd], check=True, capture_output=True, text=True, cwd=ROOT
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "ripii").rglob("*.py"))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    paths.append(ROOT / "pyproject.toml")
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "base",
            "no_renorm",
            "no_graph",
            "no_vq",
            "no_action",
            "no_geom",
            "no_scale",
            "no_identity",
            "no_spectral",
            "no_depth",
            "no_equiv",
            "no_moment",
            "plain_ae",
        ],
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=str, default="runs/ripii/benchmark.json")
    parser.add_argument("--retain-run-dir", type=str, default="")
    parser.add_argument("--study-id", type=str, default="")
    parser.add_argument("--protocol", type=str, default="")
    parser.add_argument("--protocol-sha256", type=str, default="")
    args = parser.parse_args()
    protocol_fields = (args.study_id, args.protocol, args.protocol_sha256)
    if any(protocol_fields) and not all(protocol_fields):
        parser.error(
            "--study-id, --protocol, and --protocol-sha256 must be used together"
        )
    protocol_path = Path(args.protocol) if args.protocol else None
    if protocol_path is not None:
        if protocol_path.is_symlink() or not protocol_path.is_file():
            parser.error("protocol must be a regular non-symlink file")
        actual_protocol_sha = sha256(protocol_path)
        if actual_protocol_sha != args.protocol_sha256:
            parser.error("protocol SHA-256 does not match")
        if not args.retain_run_dir:
            parser.error("a protocol-bound run requires --retain-run-dir")
    else:
        actual_protocol_sha = None
    base = load_config(args.config)
    if base.profile == "plumbing_smoke":
        parser.error(
            "plumbing_smoke disables RIPII mechanisms and cannot be used for ablations; "
            "use configs/mechanism_smoke.yaml"
        )
    out = Path(args.output)
    # A frozen study owns its output directory and keeps the conventional
    # manifest.json name. Mutable development outputs may share a directory,
    # so give each one a collision-free sidecar manifest.
    manifest_path = (
        out.with_name("manifest.json")
        if protocol_path is not None
        else out.with_suffix(".manifest.json")
    )
    companions = [out, out.with_suffix(".csv"), out.with_suffix(".md"), manifest_path]
    existing = [path for path in companions if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite benchmark output: {existing[0]}")
    retain_root = Path(args.retain_run_dir) if args.retain_run_dir else None
    if (
        protocol_path is not None
        and retain_root is not None
        and retain_root.parent.resolve() != out.parent.resolve()
    ):
        parser.error(
            "a protocol-bound retained run directory must share the output parent"
        )
    if retain_root is not None and retain_root.exists():
        parser.error(f"refusing to overwrite retained run directory: {retain_root}")
    results: list[dict[str, Any]] = []
    base_trainable_params = sum(
        parameter.numel()
        for parameter in build_model(runtime_profile(base)).parameters()
        if parameter.requires_grad
    )
    # Keep potentially large checkpoints beside the requested output. System
    # temporary volumes are often much smaller, and a full volume can surface as
    # SIGBUS inside PyTorch rather than a useful Python exception.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ripii-benchmark-", dir=out.parent
    ) as tmpdir:
        tmpdir = Path(tmpdir)
        initial_states: dict[int, Path] = {}
        for seed in args.seeds:
            init_cfg = runtime_profile(base)
            init_cfg.seed = int(seed)
            seed_everything(init_cfg.seed)
            initial_state = tmpdir / f"initial_state_{seed}.pt"
            torch.save({"model": build_model(init_cfg).state_dict()}, initial_state)
            initial_states[int(seed)] = initial_state
        if retain_root is not None:
            initial_root = retain_root / "_initial_states"
            initial_root.mkdir(parents=True, exist_ok=False)
            for seed, initial_state in initial_states.items():
                shutil.copy2(initial_state, initial_root / f"seed_{seed}.pt")
        for mode in args.modes:
            for seed in args.seeds:
                cfg = runtime_profile(apply_mode(base, mode))
                if mode == "plain_ae":
                    hidden_dim, _, _ = match_plain_autoencoder_capacity(
                        input_dim=cfg.input_dim,
                        latent_dim=cfg.latent_dim,
                        target_trainable_params=base_trainable_params,
                        reference_hidden_dim=cfg.hidden_dim,
                    )
                    cfg = replace(cfg, hidden_dim=hidden_dim)
                cfg.seed = int(seed)
                cfg.steps = int(args.steps)
                cfg.output_dir = str(tmpdir / f"{mode}_{seed}")
                cfg.compile_model = False
                cfg.amp = False
                cfg.log_every = max(1, cfg.steps)
                cfg.eval_every = max(1, cfg.steps)
                cfg_path = tmpdir / f"cfg_{mode}_{seed}.yaml"
                save_config(cfg, cfg_path)
                run(
                    [
                        "scripts/train.py",
                        "--config",
                        str(cfg_path),
                        "--mode",
                        "base",
                        "--initial-state",
                        str(initial_states[int(seed)]),
                    ]
                )
                checkpoint = Path(cfg.output_dir) / "final.pt"
                eval_proc = run(
                    [
                        "scripts/evaluate.py",
                        "--config",
                        str(cfg_path),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )
                eval_summary = json.loads(eval_proc.stdout.strip())
                run_metadata = json.loads(
                    (Path(cfg.output_dir) / "run_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                retained_path = None
                if retain_root is not None:
                    retained_path = retain_root / mode / f"seed_{seed}"
                    retained_path.mkdir(parents=True, exist_ok=False)
                    for name in (
                        "config.yaml",
                        "run_metadata.json",
                        "history.jsonl",
                        "final.pt",
                        "eval.json",
                    ):
                        shutil.copy2(Path(cfg.output_dir) / name, retained_path / name)
                results.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "initialization": run_metadata["initialization"],
                        "retained_run": str(retained_path) if retained_path else None,
                        **eval_summary,
                    }
                )
    numeric_keys = sorted(
        {
            k
            for row in results
            for k, v in row.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and k not in {"seed"}
        }
    )
    summary: dict[str, Any] = {
        "runs": results,
        "study_id": args.study_id or None,
        "evidence_status": "frozen_pilot" if protocol_path else "development_only",
        "protocol_frozen": protocol_path is not None,
        "protocol_sha256": actual_protocol_sha,
        "config_sha256": sha256(Path(args.config)),
        "profile": base.profile,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source_sha256": source_hashes(),
        "paired_design": {
            "shared_initial_state": True,
            "shared_dataset_and_splits": True,
            "shared_training_rng_reset": True,
            "parameter_matched": False,
            "comparison_type": "component-removal ablation plus plain architecture baseline",
            "cross_architecture_initialization": (
                "same seed; only shape-compatible tensors are shared with plain_ae"
            ),
        },
    }
    grouped = {}
    for row in results:
        grouped.setdefault(row["mode"], []).append(row)
    summary["by_mode"] = {}
    for mode, rows in grouped.items():
        mode_summary = {}
        for key in numeric_keys:
            vals = [
                float(r[key])
                for r in rows
                if isinstance(r.get(key), (int, float))
                and not isinstance(r.get(key), bool)
            ]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                mode_summary[f"{key}_mean"] = mean
                mode_summary[f"{key}_std"] = std
        summary["by_mode"][mode] = mode_summary
    if "base" in grouped and "plain_ae" in grouped:
        base_counts = {
            int(row["seed"]): int(row["trainable_params"]) for row in grouped["base"]
        }
        plain_counts = {
            int(row["seed"]): int(row["trainable_params"])
            for row in grouped["plain_ae"]
        }
        matched_seeds = sorted(set(base_counts) & set(plain_counts))
        relative_errors = [
            abs(plain_counts[seed] - base_counts[seed]) / base_counts[seed]
            for seed in matched_seeds
        ]
        summary["paired_design"]["parameter_match_tolerance"] = 0.02
        summary["paired_design"]["parameter_relative_errors"] = relative_errors
        summary["paired_design"]["parameter_matched"] = bool(relative_errors) and all(
            error <= 0.02 for error in relative_errors
        )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted({k for row in results for k in row.keys()})
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    report_path = out.with_suffix(".md")
    report_path.write_text(
        render_markdown(summarize_benchmark(summary)), encoding="utf-8"
    )
    artifacts = [out, csv_path, report_path]
    if retain_root is not None:
        artifacts.extend(
            sorted(path for path in retain_root.rglob("*") if path.is_file())
        )
    manifest = {
        "study_id": args.study_id or None,
        "protocol_sha256": actual_protocol_sha,
        "artifacts": [
            {
                "relative_path": str(path.relative_to(out.parent)),
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
            for path in artifacts
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
