from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from ripii.world.data import DatasetSpec, load_dataset
from ripii.world.experiment import Experiment, load_model, verify, write_json
from ripii.world.physics import Physics
from ripii.world.profiling import ProfileConfig, profile_rollout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_output(path: Path) -> dict[str, str | int | bool]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("suite profile must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = _signature(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("suite profile is invalid or non-finite JSON") from exc
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (
        payload.get("format") != "ripii-world-suite-profile-v1"
        or payload.get("signature") != expected
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("suite profile signature or schema is invalid")
    for row in rows:
        profile = row.get("profile")
        timing = profile.get("timing_seconds") if isinstance(profile, dict) else None
        values = (
            timing.get("mean") if isinstance(timing, dict) else None,
            timing.get("median") if isinstance(timing, dict) else None,
            profile.get("scene_steps_per_second_at_mean")
            if isinstance(profile, dict)
            else None,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError("suite profile contains invalid measurements")
    return {
        "status": "PASS",
        "signature": payload["signature"],
        "rows_verified": len(rows),
        "finite_values": True,
    }


def analyze(run_dir: Path, *, warmup: int = 5, repeats: int = 20) -> dict:
    run_dir = Path(run_dir)
    verification = verify(run_dir)
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cfg = Experiment(**protocol["experiment"])
    physics = Physics(**protocol["physics"])
    data, dataset_record = load_dataset(
        DatasetSpec("test", 8, 1, cfg.data_seed, cfg.max_objects, physics=physics)
    )
    rows = []
    for variant in protocol["variants"]:
        for bottleneck in protocol["bottlenecks"]:
            for seed in protocol["seeds"]:
                checkpoint = (
                    run_dir / f"{variant}_{bottleneck}" / f"seed_{seed}" / "best.pt"
                )
                model, saved = load_model(checkpoint)
                measurement = profile_rollout(
                    model,
                    data,
                    ProfileConfig(warmup, repeats, True),
                )
                rows.append(
                    {
                        "variant": variant,
                        "bottleneck": bottleneck,
                        "seed": seed,
                        "selected_step": saved["completed_steps"],
                        "parameters": measurement["trainable_parameters"],
                        "checkpoint_bytes": checkpoint.stat().st_size,
                        "checkpoint_sha256": _sha256(checkpoint),
                        "profile": measurement,
                    }
                )
    by_model = {}
    for variant in protocol["variants"]:
        selected = [row for row in rows if row["variant"] == variant]
        by_model[variant] = {
            "seeds": len(selected),
            "median_latency_ms_mean_across_seeds": sum(
                1000.0 * row["profile"]["timing_seconds"]["median"] for row in selected
            )
            / len(selected),
            "throughput_mean_across_seeds": sum(
                row["profile"]["scene_steps_per_second_at_mean"] for row in selected
            )
            / len(selected),
            "supported_flops_per_batch": sorted(
                {
                    row["profile"]["recognized_operator_flops_per_rollout"]
                    for row in selected
                }
            ),
            "parameters": sorted({row["parameters"] for row in selected}),
        }
    result = {
        "format": "ripii-world-suite-profile-v1",
        "evidence_status": "engineering_efficiency_measurement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": run_dir.name,
        "source_protocol_sha256": _sha256(protocol_path),
        "run_verification": verification,
        "dataset": dataset_record,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not_reported_by_platform",
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
        },
        "measurement_protocol": {
            "device": "cpu",
            "batch_scenes": 8,
            "trajectory_steps_per_forward": 1,
            "warmup": warmup,
            "repeats": repeats,
            "timing": "perf_counter around individual no-grad rollouts",
        },
        "source_sha256": {
            "scripts/profile_world_efficiency.py": _sha256(Path(__file__)),
            "ripii/world/profiling.py": _sha256(
                Path(__file__).resolve().parents[1] / "ripii/world/profiling.py"
            ),
        },
        "rows": rows,
        "by_model": by_model,
        "limitations": [
            "single host and process; latency is not portable",
            "profiler FLOPs omit unsupported operators",
            "CPU process peak memory is not reported",
            "inference measurements do not compute-match training",
        ],
    }
    result["signature"] = _signature(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile retained world-model forwards"
    )
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.verify_output is not None:
        if args.run_dir is not None or args.output is not None:
            parser.error("--verify-output cannot be combined with a run or output")
        print(json.dumps(verify_output(args.verify_output), indent=2))
        return
    if args.run_dir is None or args.output is None:
        parser.error("profiling requires run_dir and --output")
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    result = analyze(args.run_dir, warmup=args.warmup, repeats=args.repeats)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
