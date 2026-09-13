#!/usr/bin/env python3
"""Restart-safe launcher for the frozen RIPII 0.2 development matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ripii.utils.statistics import holm_adjust, paired_summary

SEEDS = (1009, 1013, 1019, 1021, 1031)
MODES = (
    "base",
    "simple_objective",
    "simple_plus_equiv",
    "simple_plus_inv",
    "simple_plus_scale",
    "simple_plus_proj",
    "simple_plus_spectral",
    "simple_plus_geom",
    "simple_plus_vq",
    "simple_plus_node",
    "simple_plus_moment",
    "simple_plus_identity",
    "simple_plus_depth",
    "projectors_1",
    "projectors_2",
    "projectors_4",
    "levels_0",
    "levels_1",
    "levels_2",
    "levels_3",
    "graph_topk_1",
    "graph_topk_2",
    "graph_topk_full",
    "no_vq_balance",
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
)
TARGET_STEPS = 600
CALIBRATION_STEPS = (60, 180, 300)
DEFAULT_CALIBRATION_REPEATS = 2
DEFAULT_RETRIES = 2
DEFAULT_MIN_FREE_GB = 10.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_sources(config: Path, protocol: Path) -> dict[str, str]:
    paths = sorted((ROOT / "ripii").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.sh"))
    paths += [ROOT / "pyproject.toml", ROOT / "uv.lock", config, protocol]
    return {str(path.resolve().relative_to(ROOT)): sha256(path) for path in paths}


def verify_manifest(path: Path, *, exact: bool = True) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"invalid manifest: {path}")
    root = path.parent.resolve()
    declared: set[str] = set()
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError(f"invalid manifest row: {path}")
        relative = row.get("relative_path")
        digest = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in declared
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError(f"invalid manifest row: {path}")
        declared.add(relative)
        target = path.parent / relative
        if (
            root not in target.resolve().parents
            or target.is_symlink()
            or not target.is_file()
        ):
            raise ValueError(f"missing or unsafe artifact: {target}")
        if target.stat().st_size != size or sha256(target) != digest:
            raise ValueError(f"artifact integrity failure: {target}")
    if exact:
        actual = {
            str(item.relative_to(path.parent))
            for item in path.parent.rglob("*")
            if item.is_file() and item != path
        }
        if actual != declared:
            raise ValueError(f"manifest artifact set mismatch: {path}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_directory(root: Path, path: Path) -> None:
    """Write an exact manifest for every regular, non-symlink file below root."""
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and not item.is_symlink() and item != path
    )
    write_json(
        path,
        {
            "format": "ripii-exact-directory-manifest-v1",
            "artifacts": [
                {
                    "relative_path": str(item.relative_to(root)),
                    "sha256": sha256(item),
                    "size": item.stat().st_size,
                }
                for item in files
            ],
        },
    )


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024**3)


def require_disk_space(path: Path, minimum_gib: float) -> float:
    available = free_gib(path)
    if available < minimum_gib:
        raise RuntimeError(
            f"only {available:.2f} GiB free at {path}; "
            f"at least {minimum_gib:.2f} GiB is required"
        )
    return available


def estimate_calibration(
    measurements: list[dict], *, target_steps: int = TARGET_STEPS
) -> dict:
    """Return a conservative estimate or reject timings that carry no signal."""
    grouped: dict[int, list[float]] = {}
    for row in measurements:
        steps, seconds = row.get("steps"), row.get("seconds")
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < 1
            or not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(float(seconds))
            or seconds <= 0
        ):
            raise ValueError("invalid calibration measurement")
        grouped.setdefault(steps, []).append(float(seconds))
    if (
        not isinstance(target_steps, int)
        or isinstance(target_steps, bool)
        or len(grouped) < 3
        or target_steps <= max(grouped)
    ):
        raise ValueError("calibration requires three horizons below the target")
    medians = [
        {"steps": steps, "median_seconds": statistics.median(values)}
        for steps, values in sorted(grouped.items())
    ]
    pair_slopes = [
        (right["median_seconds"] - left["median_seconds"])
        / (right["steps"] - left["steps"])
        for index, left in enumerate(medians)
        for right in medians[index + 1 :]
        if right["median_seconds"] > left["median_seconds"]
    ]
    overall_growth = medians[-1]["median_seconds"] / medians[0]["median_seconds"]
    if not pair_slopes or overall_growth < 1.05:
        raise RuntimeError(
            "calibration timings do not increase reliably with training steps; "
            "repeat calibration under stable load"
        )
    # The largest observed pairwise marginal cost and direct proportional
    # extrapolation deliberately form an upper-bound estimate.
    slope = max(pair_slopes)
    intercept = max(
        0.0,
        max(row["median_seconds"] - slope * row["steps"] for row in medians),
    )
    extrapolated = intercept + slope * target_steps
    proportional = max(
        row["median_seconds"] * target_steps / row["steps"] for row in medians
    )
    estimated_cell = max(extrapolated, proportional)
    return {
        "median_measurements": medians,
        "fixed_overhead_seconds": intercept,
        "seconds_per_step_upper_bound": slope,
        "estimated_seconds_per_target_cell": estimated_cell,
        "target_steps": target_steps,
        "estimate_method": (
            "maximum positive pairwise slope and proportional upper bound over "
            "repeated warmed measurements"
        ),
    }


def analyze(rows: list[dict]) -> dict:
    indexed = {(row["mode"], int(row["seed"])): row for row in rows}
    expected = {(mode, seed) for mode in MODES for seed in SEEDS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("study grid is missing, duplicated, or contains extra cells")
    comparisons = []
    for position, mode in enumerate(MODES[1:], start=1):
        candidate = [float(indexed[(mode, seed)]["recon"]) for seed in SEEDS]
        baseline = [float(indexed[("base", seed)]["recon"]) for seed in SEEDS]
        result = paired_summary(candidate, baseline, bootstrap_seed=20_000 + position)
        comparisons.append({"mode": mode, **result})
    adjusted = holm_adjust(
        [float(item["exact_two_sided_sign_flip_p"]) for item in comparisons]
    )
    for item, value in zip(comparisons, adjusted, strict=True):
        item["holm_adjusted_p"] = value
    controls = ("simple_objective", "plain_ae")
    improvements = {
        control: [
            1.0
            - float(indexed[("base", seed)]["recon"])
            / float(indexed[(control, seed)]["recon"])
            for seed in SEEDS
        ]
        for control in controls
    }
    decision = (
        "development_gate_pass"
        if all(value >= 0.05 for values in improvements.values() for value in values)
        else "no_advance"
    )
    return {
        "comparisons_vs_base": comparisons,
        "base_relative_improvement_vs_controls": improvements,
        "decision": decision,
        "decision_rule": (
            "base reconstruction MSE must improve by at least 5% for every paired "
            "seed versus both simple_objective and plain_ae"
        ),
        "multiplicity": "Holm adjustment across all mode-vs-base sign-flip tests",
    }


def calibrate(
    output: Path,
    config: Path,
    *,
    repeats: int = DEFAULT_CALIBRATION_REPEATS,
    workers: int = 1,
) -> dict:
    if repeats < 2 or not 1 <= workers <= len(SEEDS):
        raise ValueError("calibration requires >=2 repeats and 1..seed-count workers")
    calibration = output / "calibration.json"
    if calibration.is_file():
        cached = json.loads(calibration.read_text(encoding="utf-8"))
        if (
            cached.get("format") == "ripii-post-correction-calibration-v4"
            and cached.get("workers") == workers
        ):
            return cached
    # Warm imports, allocator state, and filesystem caches before timing.
    warmup = output / "calibration_warmup"
    if warmup.exists():
        shutil.rmtree(warmup)
    warmup.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark.py"),
            "--config",
            str(config),
            "--seeds",
            str(SEEDS[0]),
            "--modes",
            "base",
            "--steps",
            "20",
            "--output",
            str(warmup / "benchmark.json"),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    measurements = []
    for steps in CALIBRATION_STEPS:
        for repeat in range(repeats):
            directory = output / f"calibration_{steps}_r{repeat + 1}"
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
            started = time.perf_counter()
            def run_seed(
                seed: int,
                timed_steps: int = steps,
                timed_directory: Path = directory,
            ) -> None:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/benchmark.py"),
                        "--config",
                        str(config),
                        "--seeds",
                        str(seed),
                        "--modes",
                        "base",
                        "--steps",
                        str(timed_steps),
                        "--output",
                        str(timed_directory / f"seed_{seed}.json"),
                    ],
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                futures = [executor.submit(run_seed, seed) for seed in SEEDS]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            measurements.append(
                {
                    "steps": steps,
                    "repeat": repeat + 1,
                    "seconds": time.perf_counter() - started,
                    "seeds": list(SEEDS),
                    "workers": workers,
                }
            )
    estimate = estimate_calibration(measurements)
    safety_factor = 1.25
    total_seconds = (
        estimate["estimated_seconds_per_target_cell"]
        * len(MODES)
        * safety_factor
    )
    result = {
        "format": "ripii-post-correction-calibration-v4",
        "measurements": measurements,
        **estimate,
        "estimated_seconds_per_600_step_seed_batch": estimate[
            "estimated_seconds_per_target_cell"
        ],
        "estimated_total_hours": total_seconds / 3600,
        "safety_factor": safety_factor,
        "workers": workers,
        "parallelism_boundary": (
            "each timing runs the complete five-seed mode workload through the "
            "configured worker pool; the observed concurrent wall time is then "
            "extrapolated to 600 steps and guarded by the safety factor"
        ),
    }
    write_json(calibration, result)
    return result


def cell_directory(output: Path, mode: str, seed: int) -> Path:
    return output / mode / f"seed_{seed}"


def verified_cell(output: Path, mode: str, seed: int) -> bool:
    manifest = cell_directory(output, mode, seed) / "manifest.json"
    if not manifest.is_file():
        return False
    verify_manifest(manifest)
    payload = json.loads(
        (cell_directory(output, mode, seed) / "benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload.get("runs")
    return bool(
        isinstance(rows, list)
        and len(rows) == 1
        and rows[0].get("mode") == mode
        and rows[0].get("seed") == seed
    )


def safely_verified_cell(output: Path, mode: str, seed: int) -> bool:
    try:
        return verified_cell(output, mode, seed)
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return False


def quarantine(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = path.with_name(f"{path.name}.incomplete.{stamp}")
    path.rename(target)
    return target


def run_cell(
    *,
    output: Path,
    config: Path,
    protocol: Path,
    protocol_sha: str,
    mode: str,
    seed: int,
    retries: int,
    minimum_free_gib: float,
    launch: dict,
    launch_path: Path,
    event_lock=None,
) -> None:
    def record(event: dict) -> None:
        if event_lock is None:
            launch["events"].append(event)
            write_json(launch_path, launch)
            return
        with event_lock:
            launch["events"].append(event)
            write_json(launch_path, launch)

    cell = cell_directory(output, mode, seed)
    logs = output / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if safely_verified_cell(output, mode, seed):
        print(f"verified/skipped {mode} seed={seed}", flush=True)
        return
    if cell.exists() and any(cell.iterdir()):
        moved = quarantine(cell)
        record(
            {
                "mode": mode,
                "seed": seed,
                "event": "quarantined_incomplete",
                "path": str(moved),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
    elif cell.exists():
        cell.rmdir()
    for attempt in range(1, retries + 2):
        require_disk_space(output, minimum_free_gib)
        cell.mkdir(parents=True, exist_ok=False)
        log_path = logs / f"{mode}.seed_{seed}.attempt_{attempt}.log"
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmark.py"),
            "--config",
            str(config),
            "--seeds",
            str(seed),
            "--modes",
            mode,
            "--steps",
            str(TARGET_STEPS),
            "--output",
            str(cell / "benchmark.json"),
            "--retain-run-dir",
            str(cell / "runs"),
            "--study-id",
            "post_correction_v02",
            "--protocol",
            str(protocol),
            "--protocol-sha256",
            protocol_sha,
        ]
        record(
            {
                "mode": mode,
                "seed": seed,
                "attempt": attempt,
                "event": "cell_started",
                "log": str(log_path),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if process.returncode == 0:
            try:
                if not verified_cell(output, mode, seed):
                    raise ValueError("cell completed without a valid exact manifest")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                error = f"post-run integrity check failed: {exc}"
            else:
                launch["status"] = "running"
                record(
                    {
                        "mode": mode,
                        "seed": seed,
                        "attempt": attempt,
                        "event": "cell_complete",
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"completed {mode} seed={seed}", flush=True)
                return
        else:
            error = f"benchmark exited {process.returncode}; see {log_path}"
        record(
            {
                "mode": mode,
                "seed": seed,
                "attempt": attempt,
                "event": "cell_attempt_failed",
                "error": error,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        moved = quarantine(cell)
        record(
            {
                "mode": mode,
                "seed": seed,
                "attempt": attempt,
                "event": "quarantined_incomplete",
                "path": str(moved),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
    raise RuntimeError(error)


def aggregate_mode(output: Path, mode: str) -> list[dict]:
    group = output / mode
    rows: list[dict] = []
    for seed in SEEDS:
        payload = json.loads(
            (cell_directory(output, mode, seed) / "benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        cell_rows = payload.get("runs")
        if not isinstance(cell_rows, list) or len(cell_rows) != 1:
            raise ValueError(f"invalid retained cell for {mode} seed={seed}")
        rows.extend(cell_rows)
    write_json(
        group / "benchmark.json",
        {
            "format": "ripii-post-correction-mode-aggregate-v1",
            "evidence_status": "synthetic_development_only",
            "mode": mode,
            "seeds": list(SEEDS),
            "runs": rows,
        },
    )
    manifest_directory(group, group / "manifest.json")
    verify_manifest(group / "manifest.json")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/post_correction_v02"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--duration-check", action="store_true")
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--calibration-repeats", type=int, default=DEFAULT_CALIBRATION_REPEATS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument(
        "--stop-after-cells",
        type=int,
        default=0,
        help="run only this many verified cells, for a full-length sentinel check",
    )
    args = parser.parse_args()
    if (
        args.retries < 0
        or args.calibration_repeats < 2
        or args.min_free_gib <= 0
        or args.stop_after_cells < 0
        or not 1 <= args.workers <= len(SEEDS)
    ):
        parser.error("retries, calibration repeats, and minimum free space are invalid")
    output = (
        (ROOT / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve()
    )
    config = ROOT / "configs/post_correction_v02.yaml"
    protocol = ROOT / "research/protocols/post_correction_v02.md"
    snapshot = frozen_sources(config, protocol)
    launch_path = output / "launch.json"
    if args.duration_check:
        require_disk_space(output, args.min_free_gib)
        result = calibrate(
            output, config, repeats=args.calibration_repeats, workers=args.workers
        )
        if result["estimated_total_hours"] > args.max_hours:
            raise RuntimeError(
                f"estimated {result['estimated_total_hours']:.1f} hours exceeds "
                f"the {args.max_hours:.1f}-hour launch limit"
            )
        print(json.dumps({"status": "DURATION_OK", **result}, indent=2))
        return
    if args.status:
        launch = (
            json.loads(launch_path.read_text(encoding="utf-8"))
            if launch_path.is_file()
            else {"status": "not_started", "events": []}
        )
        completed_modes = len(
            {
                row.get("mode")
                for row in launch.get("events", [])
                if row.get("event") == "complete"
            }
        )
        completed_cells = sum(
            safely_verified_cell(output, mode, seed)
            for mode in MODES
            for seed in SEEDS
        )
        pid_path = output / "overnight.pid"
        pid = None
        alive = False
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                import os

                os.kill(pid, 0)
                alive = True
            except (OSError, ValueError):
                alive = False
        failed = next(
            (
                row
                for row in reversed(launch.get("events", []))
                if row.get("event") in {"failed", "cell_attempt_failed"}
            ),
            None,
        )
        print(
            json.dumps(
                {
                    "status": launch.get("status"),
                    "completed_modes": completed_modes,
                    "total_modes": len(MODES),
                    "completed_cells": completed_cells,
                    "total_cells": len(MODES) * len(SEEDS),
                    "runner_pid": pid,
                    "runner_alive": alive,
                    "last_failure": failed,
                    "free_gib": free_gib(output),
                    "calibration": json.loads(
                        (output / "calibration.json").read_text(encoding="utf-8")
                    )
                    if (output / "calibration.json").is_file()
                    else None,
                },
                indent=2,
            )
        )
        return
    if args.calibrate_only:
        output.mkdir(parents=True, exist_ok=True)
        require_disk_space(output, args.min_free_gib)
        print(
            json.dumps(
                calibrate(
                    output,
                    config,
                    repeats=args.calibration_repeats,
                    workers=args.workers,
                ),
                indent=2,
            )
        )
        return
    if args.preflight:
        available = require_disk_space(output, args.min_free_gib)
        print(
            json.dumps(
                {
                    "status": "READY",
                    "cells": len(SEEDS) * len(MODES),
                    "output": str(output),
                    "source_files": len(snapshot),
                    "free_gib": available,
                    "minimum_free_gib": args.min_free_gib,
                },
                indent=2,
            )
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    if launch_path.exists():
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if launch.get("source_sha256") != snapshot:
            raise RuntimeError(
                "source/config/protocol drift since the overnight run began"
            )
        if (
            launch.get("status") == "complete"
            and (output / "study_manifest.json").is_file()
        ):
            verify_manifest(output / "study_manifest.json", exact=False)
            for mode in MODES:
                verify_manifest(output / mode / "manifest.json")
            print(json.dumps({"status": "COMPLETE", "output": str(output)}, indent=2))
            return
    else:
        launch = {
            "format": "ripii-post-correction-launch-v2",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "seeds": list(SEEDS),
            "modes": list(MODES),
            "steps": TARGET_STEPS,
            "retries": args.retries,
            "workers": args.workers,
            "minimum_free_gib": args.min_free_gib,
            "calibration": json.loads(
                (output / "calibration.json").read_text(encoding="utf-8")
            )
            if (output / "calibration.json").is_file()
            else None,
            "events": [],
            "source_sha256": snapshot,
        }
        write_json(launch_path, launch)
    protocol_sha = sha256(protocol)
    launch["status"] = "running"
    write_json(launch_path, launch)
    processed_cells = 0
    event_lock = threading.Lock()
    for mode in MODES:
        group = output / mode
        manifest = group / "manifest.json"
        if manifest.exists():
            try:
                verify_manifest(manifest)
                print(f"verified/skipped {mode}", flush=True)
                continue
            except ValueError:
                moved = quarantine(group)
                launch["events"].append(
                    {
                        "mode": mode,
                        "event": "quarantined_invalid_mode",
                        "path": str(moved),
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                write_json(launch_path, launch)
        try:
            effective_workers = 1 if args.stop_after_cells else args.workers
            seeds_to_run = SEEDS[:1] if args.stop_after_cells else SEEDS
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=effective_workers
            ) as executor:
                futures = {
                    executor.submit(
                        run_cell,
                        output=output,
                        config=config,
                        protocol=protocol,
                        protocol_sha=protocol_sha,
                        mode=mode,
                        seed=seed,
                        retries=args.retries,
                        minimum_free_gib=args.min_free_gib,
                        launch=launch,
                        launch_path=launch_path,
                        event_lock=event_lock,
                    ): seed
                    for seed in seeds_to_run
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    processed_cells += 1
                    if args.stop_after_cells and processed_cells >= args.stop_after_cells:
                        launch["status"] = "sentinel_complete"
                        launch["events"].append(
                            {
                                "event": "sentinel_complete",
                                "verified_cells": processed_cells,
                                "at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        write_json(launch_path, launch)
                        print(
                            json.dumps(
                                {
                                    "status": "SENTINEL_COMPLETE",
                                    "verified_cells": processed_cells,
                                    "output": str(output),
                                },
                                indent=2,
                            )
                        )
                        return
            aggregate_mode(output, mode)
        except BaseException as exc:
            launch["status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            launch["events"].append(
                {
                    "mode": mode,
                    "event": launch["status"],
                    "error": str(exc)[:1000],
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json(launch_path, launch)
            raise
        launch["status"] = "running"
        launch["events"].append(
            {
                "mode": mode,
                "event": "complete",
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(launch_path, launch)
        print(f"completed {mode}", flush=True)
    rows = []
    for mode in MODES:
        payload = json.loads(
            (output / mode / "benchmark.json").read_text(encoding="utf-8")
        )
        mode_rows = payload.get("runs")
        if not isinstance(mode_rows, list) or len(mode_rows) != len(SEEDS):
            raise ValueError(f"incomplete benchmark row set for {mode}")
        rows.extend(mode_rows)
    summary_path = output / "study_summary.json"
    launch["status"] = "complete"
    launch["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(launch_path, launch)
    write_json(
        summary_path,
        {
            "format": "ripii-post-correction-summary-v1",
            "evidence_status": "synthetic_development_only",
            "protocol_sha256": protocol_sha,
            "cells": len(rows),
            "runs": rows,
            "analysis": analyze(rows),
        },
    )
    declared = [launch_path, summary_path]
    declared += [output / mode / "manifest.json" for mode in MODES]
    write_json(
        output / "study_manifest.json",
        {
            "format": "ripii-post-correction-study-manifest-v1",
            "artifacts": [
                {
                    "relative_path": str(path.relative_to(output)),
                    "sha256": sha256(path),
                    "size": path.stat().st_size,
                }
                for path in declared
            ],
        },
    )
    print(json.dumps({"status": "COMPLETE", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
