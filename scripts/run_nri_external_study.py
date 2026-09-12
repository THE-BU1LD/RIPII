#!/usr/bin/env python3
"""Run and verify the frozen NRI external-simulator development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from ripii.utils.statistics import paired_summary
from ripii.world.experiment import (
    ANALYTIC_BASELINES,
    CAPACITY_MATCH_TOLERANCE,
    Experiment,
    evaluate,
    load_model,
    train,
    widths,
    write_json,
)
from ripii.world.models import VARIANTS
from ripii.world.nri_data import DOMAINS, load_nri_dataset
from ripii.world.run_status import RunTracker, verify_complete_status

FORMAT = "ripii-nri-external-study-v1"
VARIANT_ORDER = ("mlp", "graph", "transformer", "global_pool", "multiscale")
FROZEN_SEEDS = (1009, 1013, 1019)


def frozen_experiment() -> Experiment:
    return Experiment(
        steps=100,
        train_scenes=128,
        eval_scenes=32,
        train_horizon=16,
        test_horizon=32,
        rollout_steps=4,
        batch_size=32,
        hidden=64,
        validate_every=20,
        data_seed=41_003,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "result_sha256"}
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _parse_json_object(content: str, label: str) -> dict:
    try:
        value = json.loads(content, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid retained JSON: {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"retained JSON must be an object: {label}")
    return value


def verify_run(output: Path) -> dict:
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("study manifest is missing or unsafe")
    manifest = _load_json(manifest_path)
    if manifest.get("format") != FORMAT or not isinstance(
        manifest.get("artifacts"), list
    ):
        raise ValueError("invalid study manifest schema")
    seen: set[str] = set()
    for entry in manifest["artifacts"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
            or not isinstance(entry.get("bytes"), int)
        ):
            raise ValueError("invalid study artifact entry")
        relative = entry["path"]
        path = output / relative
        if (
            relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"study artifact verification failed: {relative}")
        seen.add(relative)
    actual = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if not seen or actual != seen:
        raise ValueError("study artifact set differs from manifest")
    verify_complete_status(output / "status.json", sha256(output / "protocol.json"))
    return {"status": "PASS", "artifacts_verified": len(seen)}


def capture(output: Path, destination: Path) -> dict:
    verification = verify_run(output)
    retained = {}
    for name in (
        "protocol.json",
        "datasets.json",
        "summary.json",
        "report.md",
        "manifest.json",
    ):
        path = output / name
        retained[name] = {
            "content_text": path.read_text(encoding="utf-8"),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    payload = {
        "format": "ripii-nri-external-capsule-v1",
        "evidence_class": "external_simulator_development_evidence",
        "verification": verification,
        "retained": retained,
        "claim_boundary": (
            "Portable summaries and full-run hashes are retained; ignored checkpoints are "
            "required for model re-evaluation. This is not confirmatory or real-world evidence."
        ),
    }
    payload["result_sha256"] = _signature(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing capsule: {destination}")
    write_json(destination, payload)
    return payload


def verify_capsule(path: Path) -> dict:
    payload = _load_json(path)
    if (
        payload.get("format") != "ripii-nri-external-capsule-v1"
        or payload.get("result_sha256") != _signature(payload)
        or set(payload.get("retained", {}))
        != {
            "protocol.json",
            "datasets.json",
            "summary.json",
            "report.md",
            "manifest.json",
        }
    ):
        raise ValueError("invalid NRI external capsule or signature")
    parsed = {}
    for name, entry in payload["retained"].items():
        content = entry.get("content_text")
        if (
            not isinstance(content, str)
            or len(content.encode()) != entry.get("bytes")
            or hashlib.sha256(content.encode()).hexdigest() != entry.get("sha256")
        ):
            raise ValueError(f"capsule retained content mismatch: {name}")
        if name.endswith(".json"):
            parsed[name] = _parse_json_object(content, name)

    protocol = parsed["protocol.json"]
    datasets = parsed["datasets.json"]
    summary = parsed["summary.json"]
    manifest = parsed["manifest.json"]
    expected_cells = {
        (domain, variant, seed)
        for domain in DOMAINS
        for variant in VARIANT_ORDER
        for seed in FROZEN_SEEDS
    }
    rows = summary.get("runs")
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "frozen_development_protocol"
        or protocol.get("domains") != list(DOMAINS)
        or protocol.get("seeds") != list(FROZEN_SEEDS)
        or protocol.get("variants") != list(VARIANT_ORDER)
        or datasets.get("format") != "ripii-external-dataset-registry-v1"
        or protocol.get("datasets") != datasets.get("datasets")
        or summary.get("format") != FORMAT
        or summary.get("protocol_sha256")
        != payload["retained"]["protocol.json"]["sha256"]
        or summary.get("dataset_registry_sha256")
        != payload["retained"]["datasets.json"]["sha256"]
        or not isinstance(rows, list)
        or len(rows) != len(expected_cells)
    ):
        raise ValueError("NRI capsule protocol, dataset, or summary contract mismatch")

    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("NRI capsule contains an invalid run row")
        key = (row.get("domain"), row.get("variant"), row.get("seed"))
        metrics = row.get("metrics")
        if (
            key not in expected_cells
            or key in indexed
            or not isinstance(row.get("parameters"), int)
            or row["parameters"] < 1
            or not isinstance(row.get("selected_step"), int)
            or not 1 <= row["selected_step"] <= frozen_experiment().steps
            or not isinstance(metrics, dict)
            or set(metrics) != {"iid", "law_shift"}
        ):
            raise ValueError("NRI capsule contains a missing, duplicate, or invalid cell")
        for split_metrics in metrics.values():
            if (
                not isinstance(split_metrics, dict)
                or not split_metrics
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in split_metrics.values()
                )
            ):
                raise ValueError("NRI capsule contains invalid or non-finite metrics")
        indexed[key] = row
    if set(indexed) != expected_cells:
        raise ValueError("NRI capsule cell grid is incomplete")

    relative = [
        1.0
        - indexed[(domain, "multiscale", seed)]["metrics"]["iid"]["position_rmse"]
        / indexed[(domain, "global_pool", seed)]["metrics"]["iid"]["position_rmse"]
        for domain in DOMAINS
        for seed in FROZEN_SEEDS
    ]
    stored_relative = summary.get(
        "multiscale_vs_global_pool_iid_relative_improvements"
    )
    expected_decision = (
        "development_gate_pass"
        if all(value >= 0.05 for value in relative)
        else "no_advance"
    )
    if (
        not isinstance(stored_relative, list)
        or len(stored_relative) != len(relative)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in stored_relative
        )
        or any(abs(left - right) > 1e-12 for left, right in zip(relative, stored_relative))
        or summary.get("decision") != expected_decision
        or f"Decision: **{expected_decision}**."
        not in payload["retained"]["report.md"]["content_text"]
    ):
        raise ValueError("NRI capsule decision is inconsistent with retained metrics")

    artifacts = manifest.get("artifacts")
    if (
        manifest.get("format") != FORMAT
        or not isinstance(artifacts, list)
        or payload.get("verification", {}).get("status") != "PASS"
        or payload.get("verification", {}).get("artifacts_verified") != len(artifacts)
    ):
        raise ValueError("NRI capsule manifest or verification record is invalid")
    declared = {entry.get("path"): entry for entry in artifacts if isinstance(entry, dict)}
    if len(declared) != len(artifacts):
        raise ValueError("NRI capsule manifest has duplicate or invalid entries")
    for name in ("protocol.json", "datasets.json", "summary.json", "report.md"):
        entry = declared.get(name)
        retained = payload["retained"][name]
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != retained["sha256"]
            or entry.get("bytes") != retained["bytes"]
        ):
            raise ValueError(f"NRI capsule manifest mismatch: {name}")
    return {
        "status": "PASS",
        "retained_files": 5,
        "result_sha256": payload["result_sha256"],
    }


def _protocol(
    cfg: Experiment, data_root: Path, seeds: tuple[int, ...]
) -> tuple[dict, dict]:
    capacity = widths(cfg, VARIANT_ORDER, "continuous")
    if any(
        row["relative_error"] > CAPACITY_MATCH_TOLERANCE for row in capacity.values()
    ):
        raise ValueError("external study parameter matching exceeds 5%")
    datasets = {}
    for domain in DOMAINS:
        for split, scenes, horizon in (
            ("train", cfg.train_scenes, cfg.train_horizon),
            ("validation", cfg.eval_scenes, cfg.rollout_steps),
            ("test", cfg.eval_scenes, cfg.test_horizon),
        ):
            _, record = load_nri_dataset(
                data_root, domain, split, scenes, horizon, cfg.max_objects
            )
            datasets[f"{domain}_{split}"] = record
    source_root = Path(__file__).parents[1]
    sources = {
        str(path.relative_to(source_root)): sha256(path)
        for path in (
            source_root / "scripts/run_nri_external_study.py",
            source_root / "ripii/world/nri_data.py",
            source_root / "ripii/world/experiment.py",
            source_root / "ripii/world/models.py",
            source_root / "ripii/utils/statistics.py",
            source_root / "research/protocols/nri_external_development_v1.md",
            source_root / "pyproject.toml",
            source_root / "uv.lock",
        )
    }
    protocol = {
        "format": FORMAT,
        "status": "frozen_development_protocol",
        "evidence_class": "external_simulator_development_evidence",
        "hypothesis": "multiscale improves IID position RMSE by at least 5% versus global_pool for every domain-seed pair",
        "experiment": asdict(cfg),
        "domains": list(DOMAINS),
        "seeds": list(seeds),
        "variants": list(VARIANT_ORDER),
        "capacity": capacity,
        "capacity_tolerance": CAPACITY_MATCH_TOLERANCE,
        "selection": "minimum validation position RMSE + 0.25 * velocity RMSE",
        "evaluation": ["iid", "law_shift"],
        "primary_metric": "full-rollout position_rmse",
        "decision_rule": "all six IID domain-seed relative improvements of multiscale over global_pool are >=0.05",
        "failure_rule": "any failed, missing, duplicate, or non-finite cell invalidates the study",
        "statistics": "paired seeds within domain; fixed-case mean across the two named domains; bootstrap and exact sign-flip are descriptive at n=3",
        "compute_boundary": "parameter- and update-matched, not compute-matched",
        "datasets": datasets,
        "source_sha256": sources,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
        },
    }
    return protocol, datasets


def _run_impl(
    data_root: Path, output: Path, cfg: Experiment, seeds: tuple[int, ...]
) -> dict:
    cfg.validate()
    if cfg != frozen_experiment() or seeds != FROZEN_SEEDS:
        raise ValueError(
            "configuration differs from the frozen NRI development protocol"
        )
    if tuple(VARIANT_ORDER) != tuple(VARIANTS):
        raise ValueError("model registry drifted from the frozen external protocol")
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(seed < 0 for seed in seeds):
        raise ValueError(
            "development protocol requires three distinct nonnegative seeds"
        )
    if output.exists():
        raise ValueError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    protocol, dataset_records = _protocol(cfg, data_root, seeds)
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "datasets.json",
        {"format": "ripii-external-dataset-registry-v1", "datasets": dataset_records},
    )
    tracker = RunTracker.create(
        output,
        run_kind="nri_external_development",
        protocol_sha256=sha256(output / "protocol.json"),
    )
    tracker.transition("running")
    loaded = {
        (domain, split): load_nri_dataset(
            data_root,
            domain,
            split,
            cfg.train_scenes if split == "train" else cfg.eval_scenes,
            cfg.train_horizon
            if split == "train"
            else (cfg.rollout_steps if split == "validation" else cfg.test_horizon),
            cfg.max_objects,
        )
        for domain in DOMAINS
        for split in ("train", "validation", "test")
    }
    rows = []
    for domain in DOMAINS:
        other = next(item for item in DOMAINS if item != domain)
        for variant in VARIANT_ORDER:
            width = protocol["capacity"][variant]["hidden"]
            for seed in seeds:
                directory = output / domain / variant / f"seed_{seed}"
                print(
                    f"training domain={domain} variant={variant} seed={seed}",
                    flush=True,
                )
                _, checkpoint = train(
                    cfg,
                    directory,
                    variant=variant,
                    seed=seed,
                    hidden=width,
                    prepared_datasets=(
                        loaded[(domain, "train")],
                        loaded[(domain, "validation")],
                    ),
                )
                model, selected = load_model(directory / "best.pt")
                final = torch.load(
                    directory / "final.pt", map_location="cpu", weights_only=True
                )
                row = {
                    "domain": domain,
                    "variant": variant,
                    "seed": seed,
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "selected_step": selected["completed_steps"],
                    "train_seconds": final["train_seconds"],
                    "metrics": {
                        "iid": evaluate(model, loaded[(domain, "test")][0]),
                        "law_shift": evaluate(model, loaded[(other, "test")][0]),
                    },
                }
                if not all(
                    math.isfinite(value)
                    for split in row["metrics"].values()
                    for value in split.values()
                ):
                    raise FloatingPointError(
                        "external study produced non-finite metrics"
                    )
                write_json(directory / "evaluation.json", row)
                rows.append(row)
    analytic = {
        domain: {
            name: evaluate(factory(), loaded[(domain, "test")][0])
            for name, factory in ANALYTIC_BASELINES.items()
        }
        for domain in DOMAINS
    }
    indexed = {(row["domain"], row["variant"], row["seed"]): row for row in rows}
    comparisons = {}
    for baseline in ("mlp", "graph", "transformer", "global_pool"):
        by_domain = {}
        fixed_candidate, fixed_baseline = [], []
        for offset, domain in enumerate(DOMAINS):
            candidate = [
                indexed[(domain, "multiscale", seed)]["metrics"]["iid"]["position_rmse"]
                for seed in seeds
            ]
            control = [
                indexed[(domain, baseline, seed)]["metrics"]["iid"]["position_rmse"]
                for seed in seeds
            ]
            by_domain[domain] = paired_summary(
                candidate, control, bootstrap_seed=71_000 + offset
            )
        for seed in seeds:
            fixed_candidate.append(
                sum(
                    indexed[(domain, "multiscale", seed)]["metrics"]["iid"][
                        "position_rmse"
                    ]
                    for domain in DOMAINS
                )
                / len(DOMAINS)
            )
            fixed_baseline.append(
                sum(
                    indexed[(domain, baseline, seed)]["metrics"]["iid"]["position_rmse"]
                    for domain in DOMAINS
                )
                / len(DOMAINS)
            )
        comparisons[baseline] = {
            "by_fixed_domain": by_domain,
            "fixed_case_mean_over_domains": paired_summary(
                fixed_candidate, fixed_baseline, bootstrap_seed=72_000
            ),
        }
    relative = [
        1.0
        - indexed[(domain, "multiscale", seed)]["metrics"]["iid"]["position_rmse"]
        / indexed[(domain, "global_pool", seed)]["metrics"]["iid"]["position_rmse"]
        for domain in DOMAINS
        for seed in seeds
    ]
    decision = (
        "development_gate_pass"
        if all(value >= 0.05 for value in relative)
        else "no_advance"
    )
    summary = {
        "format": FORMAT,
        "evidence_class": "external_simulator_development_evidence",
        "protocol_sha256": sha256(output / "protocol.json"),
        "dataset_registry_sha256": sha256(output / "datasets.json"),
        "runs": rows,
        "analytic_baselines": analytic,
        "comparisons": comparisons,
        "multiscale_vs_global_pool_iid_relative_improvements": relative,
        "decision": decision,
        "claim_boundary": "Three seeds on two locally generated external simulators; not powered, confirmatory, real-world, or compute-matched evidence.",
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# NRI external-simulator development study",
        "",
        f"Decision: **{decision}**.",
        "",
        "The primary six paired relative improvements (positive favors multiscale) were:",
        "",
        json.dumps(relative, indent=2),
        "",
        "This is development evidence from external simulator code, not real-world or confirmatory validation.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tracker.transition("complete")
    artifacts = [
        {
            "path": str(path.relative_to(output)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    write_json(output / "manifest.json", {"format": FORMAT, "artifacts": artifacts})
    verify_run(output)
    return summary


def run(data_root: Path, output: Path, cfg: Experiment, seeds: tuple[int, ...]) -> dict:
    """Run all cells or leave a typed failed state; never emit a partial manifest."""
    output = Path(output)
    status_existed = (output / "status.json").exists()
    try:
        return _run_impl(data_root, output, cfg, seeds)
    except BaseException as exc:
        status = output / "status.json"
        if not status_existed and status.is_file() and not status.is_symlink():
            tracker = RunTracker.open(output)
            if tracker.payload["state"] != "failed":
                tracker.transition("failed", error=exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/external/nri"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verify-capsule", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_run(args.verify), indent=2))
        return
    if args.verify_capsule:
        print(json.dumps(verify_capsule(args.verify_capsule), indent=2))
        return
    if args.output is None:
        raise SystemExit("--output is required to run or capture")
    if args.capsule:
        print(json.dumps(capture(args.output, args.capsule), indent=2)[:1000])
        return
    torch.set_num_threads(1)
    result = run(args.data, args.output, frozen_experiment(), FROZEN_SEEDS)
    print(
        json.dumps(
            {"decision": result["decision"], "runs": len(result["runs"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
