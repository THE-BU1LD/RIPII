#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from ripii.utils.statistics import descriptive_summary, paired_sign_flip_test
from ripii.world.experiment import Experiment, benchmark, verify, write_json
from ripii.world.run_status import RunTracker

VARIANTS = ("ripii_mr", "equivariant", "graph", "global_pool", "multiscale")
MODEL_SEEDS = (131, 137, 139, 149, 151)
DATA_SEEDS = (3101, 3203, 3307)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATHS = (
    "scripts/run_ripii_mr_matrix.py",
    "ripii/utils/statistics.py",
    "research/protocols/ripii_mr_development_v1.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol(args: argparse.Namespace, cfg: Experiment) -> dict:
    return {
        "format": "ripii-mr-development-matrix-v1",
        "status": "frozen_development_only",
        "hypothesis": (
            "RIPII-MR improves mean OOD position RMSE by at least 5% against every "
            "included learned control on every paired model/data seed, without more "
            "than 5% IID regression."
        ),
        "candidate": "ripii_mr",
        "controls": list(VARIANTS[1:]),
        "model_seeds": list(args.model_seeds),
        "data_seeds": list(args.data_seeds),
        "experiment": asdict(cfg),
        "selection": "validation position RMSE + 0.25 * validation velocity RMSE",
        "primary_endpoint": (
            "mean position RMSE across more_objects, composition, and fast splits"
        ),
        "decision_rule": (
            "advance_development only if every candidate-control paired data/model-seed "
            "comparison reaches >=5% OOD improvement and <=5% IID regression"
        ),
        "evidence_boundary": (
            "synthetic development only; this protocol cannot create a confirmatory, "
            "external-validity, novelty, or publication claim"
        ),
    }


def _load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def analyze(
    output: Path,
    data_seeds: tuple[int, ...],
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
) -> dict:
    rows = []
    seen_cells: set[tuple[int, int, str]] = set()
    for data_seed in data_seeds:
        summary = _load_object(output / f"data_seed_{data_seed}" / "summary.json")
        if summary.get("evidence_status") != "development_only":
            raise ValueError("child benchmark has an invalid evidence boundary")
        for row in summary.get("paired_comparisons", []):
            if row.get("candidate") != "ripii_mr":
                raise ValueError("child benchmark did not evaluate RIPII-MR")
            seed = row.get("seed")
            control = row.get("control")
            cell = (data_seed, seed, control)
            if (
                seed not in model_seeds
                or control not in VARIANTS[1:]
                or cell in seen_cells
            ):
                raise ValueError(f"invalid or duplicate paired cell: {cell}")
            improvement = row.get("ood_relative_improvement")
            regression = row.get("id_relative_regression")
            if (
                not isinstance(improvement, (int, float))
                or not isinstance(regression, (int, float))
                or not math.isfinite(improvement)
                or not math.isfinite(regression)
            ):
                raise ValueError(f"nonfinite paired cell: {cell}")
            seen_cells.add(cell)
            rows.append(
                {
                    "data_seed": data_seed,
                    **row,
                    "passes": improvement >= 0.05 and regression <= 0.05,
                }
            )
    expected_cells = {
        (data_seed, model_seed, control)
        for data_seed in data_seeds
        for model_seed in model_seeds
        for control in VARIANTS[1:]
    }
    expected = len(expected_cells)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} paired comparisons, found {len(rows)}")
    if seen_cells != expected_cells:
        raise ValueError("paired comparison cells do not match the frozen matrix")
    controls = {}
    for control in VARIANTS[1:]:
        control_rows = [row for row in rows if row["control"] == control]
        values = [row["ood_relative_improvement"] for row in control_rows]
        by_data_seed = {
            str(seed): sum(
                row["ood_relative_improvement"]
                for row in control_rows
                if row["data_seed"] == seed
            )
            / sum(row["data_seed"] == seed for row in control_rows)
            for seed in data_seeds
        }
        controls[control] = {
            "paired_cells": len(values),
            "ood_relative_improvement": descriptive_summary(values),
            "data_seed_means": by_data_seed,
            "model_seed_sign_flip_diagnostic": paired_sign_flip_test(
                values, [0.0] * len(values)
            ),
            "all_cells_pass": all(row["passes"] for row in control_rows),
        }
    decision = (
        "advance_development"
        if rows and all(row["passes"] for row in rows)
        else "no_advance"
    )
    return {
        "format": "ripii-mr-development-analysis-v1",
        "evidence_status": "synthetic_development_only",
        "decision": decision,
        "paired_comparisons": rows,
        "by_control": controls,
        "caveat": (
            "model-seed sign-flip calculations treat model/data cells descriptively; "
            "the three independent data-seed means are the appropriate cluster-level "
            "view and are not powered confirmation"
        ),
    }


def _write_report(output: Path, analysis: dict) -> None:
    lines = [
        "# RIPII-MR development matrix",
        "",
        f"Decision: **{analysis['decision']}**.",
        "",
        "Synthetic development evidence only. Negative and failed runs are retained.",
        "",
        "| Control | Paired cells | Mean OOD relative improvement | All cells pass |",
        "|---|---:|---:|---:|",
    ]
    for control, result in analysis["by_control"].items():
        lines.append(
            f"| {control} | {result['paired_cells']} | "
            f"{result['ood_relative_improvement']['mean']:.4f} | "
            f"{result['all_cells_pass']} |"
        )
    lines.extend(
        [
            "",
            (
                "The advancement rule is deliberately strict: RIPII-MR must beat every "
                "included control by at least 5% on every paired data/model seed without "
                "more than 5% IID regression."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_sha256(source_root: Path) -> dict[str, str]:
    sources = {}
    for relative in PROVENANCE_PATHS:
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing provenance source: {relative}")
        sources[relative] = sha256_file(path)
    return sources


def _manifest(output: Path, *, source_root: Path = SOURCE_ROOT) -> None:
    paths = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and not path.is_symlink()
    ]
    protocol_path = output / "protocol.json"
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise FileNotFoundError("missing matrix protocol.json")
    write_json(
        output / "manifest.json",
        {
            "format": "ripii-mr-development-manifest-v2",
            "protocol_sha256": sha256_file(protocol_path),
            "source_sha256": _source_sha256(source_root),
            "artifacts": [
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        },
    )


def verify_matrix(output: Path, *, source_root: Path = SOURCE_ROOT) -> dict:
    manifest = _load_object(output / "manifest.json")
    manifest_format = manifest.get("format")
    if manifest_format not in {
        "ripii-mr-development-manifest-v1",
        "ripii-mr-development-manifest-v2",
    }:
        raise ValueError("invalid RIPII-MR matrix manifest")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("empty RIPII-MR matrix manifest")
    root = output.resolve()
    seen = set()
    for entry in entries:
        relative = entry.get("path")
        path = (root / relative).resolve() if isinstance(relative, str) else root
        if (
            not isinstance(relative, str)
            or relative in seen
            or root not in path.parents
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"invalid or changed artifact: {relative}")
        seen.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and not path.is_symlink()
    }
    if actual != seen:
        raise ValueError(
            f"unexpected or unmanifested artifacts: {sorted(actual - seen)}"
        )
    if manifest_format == "ripii-mr-development-manifest-v2":
        protocol_path = output / "protocol.json"
        if manifest.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("changed matrix protocol")
        recorded_sources = manifest.get("source_sha256")
        if not isinstance(recorded_sources, dict) or set(recorded_sources) != set(
            PROVENANCE_PATHS
        ):
            raise ValueError("invalid matrix source provenance")
        current_sources = _source_sha256(source_root)
        changed_sources = [
            relative
            for relative in PROVENANCE_PATHS
            if recorded_sources.get(relative) != current_sources[relative]
        ]
        if changed_sources:
            raise ValueError(f"changed matrix source: {changed_sources}")
    summary = _load_object(output / "summary.json")
    if summary.get("decision") not in {"advance_development", "no_advance"}:
        raise ValueError("missing predefined decision")
    return {
        "status": "PASS",
        "decision": summary["decision"],
        "artifacts": len(entries),
    }


def run(args: argparse.Namespace) -> dict:
    if (
        len(set(args.model_seeds)) != len(args.model_seeds)
        or len(set(args.data_seeds)) != len(args.data_seeds)
        or len(args.model_seeds) != 5
        or len(args.data_seeds) != 3
        or any(seed < 0 for seed in (*args.model_seeds, *args.data_seeds))
    ):
        raise ValueError(
            "the frozen development matrix requires 5 model and 3 data seeds"
        )
    cfg = Experiment(
        steps=args.steps,
        train_scenes=args.train_scenes,
        eval_scenes=args.eval_scenes,
        train_horizon=args.train_horizon,
        test_horizon=args.test_horizon,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        hidden=args.hidden,
        max_objects=args.max_objects,
        lr=args.lr,
        validate_every=args.validate_every,
        global_coupling=args.global_coupling,
        state_noise_std=args.state_noise_std,
        rollout_curriculum_steps=args.rollout_curriculum_steps,
    )
    cfg.validate()
    output = args.output
    protocol = _protocol(args, cfg)
    if output.exists():
        existing = _load_object(output / "protocol.json")
        if existing != protocol:
            raise ValueError("existing output uses a different frozen protocol")
        tracker = RunTracker.open(output)
        if tracker.payload["state"] == "complete":
            return verify_matrix(output)
        if tracker.payload["state"] == "failed":
            raise ValueError("failed matrix output is immutable; choose a new output")
    else:
        output.mkdir(parents=True)
        write_json(output / "protocol.json", protocol)
        tracker = RunTracker.create(
            output,
            run_kind="ripii_mr_development_matrix",
            protocol_sha256=sha256_file(output / "protocol.json"),
        )
        tracker.transition("running")
    try:
        for data_seed in args.data_seeds:
            child = output / f"data_seed_{data_seed}"
            if child.exists():
                verify(child)
                continue
            child_cfg = Experiment(**{**asdict(cfg), "data_seed": data_seed})
            benchmark(
                child_cfg,
                child,
                seeds=tuple(args.model_seeds),
                variants=VARIANTS,
                bottlenecks=("continuous",),
            )
            verify(child)
        analysis = analyze(output, tuple(args.data_seeds), tuple(args.model_seeds))
        write_json(output / "summary.json", analysis)
        _write_report(output, analysis)
        tracker.transition("complete")
        _manifest(output)
        return verify_matrix(output)
    except BaseException as exc:
        if tracker.payload["state"] == "running":
            tracker.transition("failed", error=exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen RIPII-MR matrix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--train-scenes", type=int, default=384)
    parser.add_argument("--eval-scenes", type=int, default=96)
    parser.add_argument("--train-horizon", type=int, default=24)
    parser.add_argument("--test-horizon", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--global-coupling", type=float, default=1.0)
    parser.add_argument("--state-noise-std", type=float, default=0.002)
    parser.add_argument("--rollout-curriculum-steps", type=int, default=600)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    parser.add_argument("--data-seeds", type=int, nargs="+", default=list(DATA_SEEDS))
    args = parser.parse_args()
    result = verify_matrix(args.output) if args.verify_only else run(args)
    if not all(
        not isinstance(value, float) or math.isfinite(value)
        for value in result.values()
    ):
        raise FloatingPointError("non-finite runner result")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
