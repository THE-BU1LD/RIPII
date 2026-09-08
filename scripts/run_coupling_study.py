from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from ripii.utils.statistics import exact_sign_flip_pvalue, paired_summary
from ripii.world.experiment import Experiment, benchmark, write_json
from ripii.world.protocol import ExperimentProtocol
from ripii.world.run_status import RunTracker, verify_complete_status


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _finite_json(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))


def _parse_json_object(content: str, label: str) -> dict:
    try:
        payload = json.loads(content, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict) or not _finite_json(payload):
        raise ValueError(f"{label} must be a finite JSON object")
    return payload


def _load_json(path: Path) -> dict:
    return _parse_json_object(path.read_text(encoding="utf-8"), str(path))


def verify(directory: Path) -> dict[str, str | int]:
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("study manifest must be a regular non-symlink file")
    manifest = _load_json(manifest_path)
    if manifest.get("format") != "ripii-coupling-study-v1" or not isinstance(
        manifest.get("artifacts"), list
    ):
        raise ValueError("invalid coupling-study manifest schema")
    declared = {}
    for row in manifest["artifacts"]:
        relative = Path(row["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) in declared
        ):
            raise ValueError("unsafe or duplicate manifest path")
        declared[str(relative)] = row
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(declared):
        raise ValueError("coupling-study artifact set differs from manifest")
    for relative, row in declared.items():
        path = directory / relative
        if path.is_symlink() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"invalid size or symlink: {relative}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"hash mismatch: {relative}")
        if path.suffix == ".json":
            _load_json(path)
    if "status.json" in declared:
        verify_complete_status(
            directory / "status.json", _sha256(directory / "protocol.json")
        )
    return {
        "status": "PASS",
        "artifacts_verified": len(declared),
        "manifest_sha256": _sha256(manifest_path),
    }


def _capsule_signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    artifacts = [
        {
            "path": str(path.relative_to(directory)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest = {"format": "ripii-coupling-study-v1", "artifacts": artifacts}
    write_json(manifest_path, manifest)
    return manifest


def capture(directory: Path, output: Path) -> dict:
    verified = verify(directory)
    if output.exists():
        raise FileExistsError(f"refusing existing capsule: {output}")
    retained_paths = (
        "protocol.json",
        "summary.json",
        "report.md",
        "manifest.json",
        "local/summary.json",
        "coupled/summary.json",
    )
    retained = {}
    for relative in retained_paths:
        path = directory / relative
        content = path.read_text(encoding="utf-8")
        retained[relative] = {
            "content_text": content,
            "bytes": len(content.encode()),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    payload = {
        "format": "ripii-coupling-study-capsule-v1",
        "evidence_status": "development_only",
        "full_run_manifest_sha256": verified["manifest_sha256"],
        "full_run_artifacts_verified_before_capture": verified["artifacts_verified"],
        "retained": retained,
    }
    payload["signature"] = _capsule_signature(payload)
    write_json(output, payload)
    return payload


def verify_capsule(path: Path) -> dict[str, str | int | bool]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("capsule must be a regular non-symlink file")
    payload = _load_json(path)
    if payload.get("format") != "ripii-coupling-study-capsule-v1":
        raise ValueError("invalid coupling-study capsule format")
    if payload.get("signature") != _capsule_signature(payload):
        raise ValueError("coupling-study capsule signature mismatch")
    if payload.get("evidence_status") != "development_only":
        raise ValueError("invalid coupling-study evidence status")
    retained = payload.get("retained")
    expected_retained = {
        "protocol.json",
        "summary.json",
        "report.md",
        "manifest.json",
        "local/summary.json",
        "coupled/summary.json",
    }
    if not isinstance(retained, dict) or set(retained) != expected_retained:
        raise ValueError("coupling-study capsule retained-file set is invalid")
    parsed = {}
    for relative, row in retained.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid retained record: {relative}")
        content = row.get("content_text")
        if not isinstance(content, str):
            raise ValueError(f"invalid retained content: {relative}")
        encoded = content.encode()
        if len(encoded) != row.get("bytes") or hashlib.sha256(
            encoded
        ).hexdigest() != row.get("sha256"):
            raise ValueError(f"retained content verification failed: {relative}")
        if relative.endswith(".json"):
            parsed[relative] = _parse_json_object(
                content, f"retained content {relative}"
            )
    if (
        retained["manifest.json"]["sha256"]
        != payload.get("full_run_manifest_sha256")
        or parsed["manifest.json"].get("format") != "ripii-coupling-study-v1"
        or parsed["summary.json"].get("protocol_sha256")
        != retained["protocol.json"]["sha256"]
    ):
        raise ValueError("coupling-study capsule provenance links are invalid")
    return {
        "status": "PASS",
        "signature": payload["signature"],
        "retained_files": len(retained),
        "full_run_artifacts_verified_before_capture": payload[
            "full_run_artifacts_verified_before_capture"
        ],
        "content_hashes_verified": True,
    }


def _paired_rows(
    summary: dict, candidate: str, baseline: str
) -> list[tuple[int, dict, dict]]:
    raw_rows = summary.get("runs")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("summary runs must be a nonempty list")
    rows: dict[tuple[str, int], dict] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("every summary row must be an object")
        variant, seed = row.get("variant"), row.get("seed")
        if (
            not isinstance(variant, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise ValueError("summary rows require a variant and integer seed")
        key = (variant, seed)
        if key in rows:
            raise ValueError(f"duplicate result cell: {key}")
        rows[key] = row
    candidate_seeds = {seed for variant, seed in rows if variant == candidate}
    baseline_seeds = {seed for variant, seed in rows if variant == baseline}
    if not candidate_seeds or candidate_seeds != baseline_seeds:
        raise ValueError(
            f"{candidate} and {baseline} do not have the same paired seeds"
        )
    return [
        (seed, rows[(candidate, seed)], rows[(baseline, seed)])
        for seed in sorted(candidate_seeds)
    ]


def analyze(regimes: dict[str, dict]) -> dict:
    if set(regimes) != {"local", "coupled"}:
        raise ValueError("analysis requires exactly local and coupled regimes")
    comparisons: dict[str, dict] = {}
    relative_by_regime: dict[str, list[float]] = {}
    seeds_by_regime: dict[str, list[int]] = {}
    for name in ("local", "coupled"):
        summary = regimes[name]
        pairs = _paired_rows(summary, "multiscale", "global_pool")
        seeds_by_regime[name] = [row[0] for row in pairs]
        candidate = [
            float(row[1]["metrics"]["more_objects"]["position_rmse"]) for row in pairs
        ]
        baseline = [
            float(row[2]["metrics"]["more_objects"]["position_rmse"]) for row in pairs
        ]
        if not all(
            math.isfinite(value) and value >= 0 for value in candidate
        ) or not all(math.isfinite(value) and value > 0 for value in baseline):
            raise FloatingPointError(
                "paired RMSE values must be finite and controls positive"
            )
        relative = [1.0 - left / right for left, right in zip(candidate, baseline)]
        relative_by_regime[name] = relative
        comparisons[name] = {
            "multiscale_vs_global_pool_more_objects": paired_summary(
                candidate,
                baseline,
                bootstrap_seed=31_000 + len(comparisons),
            ),
            "relative_improvement_by_seed": relative,
        }
    if seeds_by_regime["local"] != seeds_by_regime["coupled"]:
        raise ValueError("local and coupled regimes do not have the same paired seeds")
    interaction = [
        coupled - local
        for coupled, local in zip(
            relative_by_regime["coupled"], relative_by_regime["local"]
        )
    ]
    coupled = relative_by_regime["coupled"]
    passes = all(value >= 0.05 for value in coupled) and all(
        value >= 0.05 for value in interaction
    )
    return {
        "primary_metric": "more_objects position RMSE",
        "experimental_unit": "paired initialization/minibatch seed",
        "paired_seeds": seeds_by_regime["local"],
        "comparisons": comparisons,
        "coupling_interaction_relative_improvement": interaction,
        "coupling_interaction_mean": sum(interaction) / len(interaction),
        "coupling_interaction_exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(
            interaction
        ),
        "decision": "advance_conditional_hierarchy" if passes else "no_advance",
        "evidence_status": "development_only",
        "claim_boundary": (
            "One synthetic force-law intervention; bootstrap intervals describe the "
            "executed seeds and are not population confidence intervals."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen local-vs-long-range hierarchy development study."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--capture", type=Path)
    action.add_argument("--verify-capsule", type=Path)
    parser.add_argument("--capsule-output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[53, 59, 61])
    parser.add_argument("--data-seed", type=int, default=12031)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=44)
    parser.add_argument("--train-scenes", type=int, default=96)
    parser.add_argument("--eval-scenes", type=int, default=16)
    args = parser.parse_args()
    if args.verify is not None:
        print(json.dumps(verify(args.verify), indent=2))
        return
    if args.verify_capsule is not None:
        print(json.dumps(verify_capsule(args.verify_capsule), indent=2))
        return
    if args.capture is not None:
        if args.capsule_output is None:
            parser.error("--capture requires --capsule-output")
        payload = capture(args.capture, args.capsule_output)
        print(
            json.dumps(
                {
                    "output": str(args.capsule_output),
                    "signature": payload["signature"],
                },
                indent=2,
            )
        )
        return
    if args.capsule_output is not None:
        parser.error("--capsule-output is valid only with --capture")
    assert args.output is not None
    if (
        args.steps < 1
        or args.coupling <= 0
        or not math.isfinite(args.coupling)
        or len(args.seeds) < 2
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError(
            "require positive steps/coupling and at least two unique seeds"
        )
    base = Experiment(
        steps=args.steps,
        train_scenes=args.train_scenes,
        eval_scenes=args.eval_scenes,
        hidden=args.hidden,
        data_seed=args.data_seed,
        validate_every=max(1, min(25, args.steps)),
    )
    protocol = {
        "format": "ripii-coupling-protocol-v1",
        "study": "RIPII conditional hierarchy coupling study v1",
        "status": "prospectively_written_development_protocol",
        "hypothesis": (
            "Learned hierarchy is useful only when nonlocal interactions require "
            "structured global communication beyond a pooled global context."
        ),
        "h0": (
            "Coupling does not give multiscale a >=5 percentage-point paired advantage "
            "over the global-pool control on larger-object scenes."
        ),
        "h1": (
            "For every seed, coupled-regime multiscale improves more-objects position "
            "RMSE over global-pool by >=5%, and that relative improvement exceeds its "
            "local-regime value by >=5 percentage points."
        ),
        "decision_rule": "all paired seed conditions in H1 must pass",
        "models": ["graph", "global_pool", "multiscale"],
        "bottleneck": "continuous",
        "seeds": args.seeds,
        "regimes": {"local": 0.0, "coupled": args.coupling},
        "experiment_without_coupling": asdict(base),
        "git": _git_state(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "ripii/world/experiment.py",
                Path(__file__).resolve().parents[1] / "ripii/world/models.py",
                Path(__file__).resolve().parents[1] / "ripii/world/physics.py",
                Path(__file__).resolve().parents[1] / "ripii/world/protocol.py",
                Path(__file__).resolve().parents[1] / "ripii/utils/statistics.py",
            )
        },
        "limitations": [
            "development study, not externally preregistered",
            "one synthetic simulator and one data seed",
            "parameter matched but not FLOP matched",
            "small seed count gives low-resolution inference",
        ],
    }
    protocol_record = ExperimentProtocol.from_mapping(protocol)
    protocol = protocol_record.as_dict()
    base = Experiment(**protocol["experiment_without_coupling"])
    if args.output.exists():
        raise FileExistsError(f"refusing existing study directory: {args.output}")
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", protocol)
    tracker = RunTracker.create(
        args.output,
        run_kind="coupling_study",
        protocol_sha256=_sha256(args.output / "protocol.json"),
    )
    tracker.transition("running")
    try:
        regimes = {}
        for name, coupling in protocol["regimes"].items():
            cfg = Experiment(**{**asdict(base), "global_coupling": coupling})
            regimes[name] = benchmark(
                cfg,
                args.output / name,
                seeds=protocol["seeds"],
                variants=protocol["models"],
                bottlenecks=[protocol["bottleneck"]],
            )
        result = analyze(regimes)
        result["protocol_sha256"] = _sha256(args.output / "protocol.json")
        write_json(args.output / "summary.json", result)
        lines = [
            "# Conditional hierarchy coupling study",
            "",
            f"Decision: **{result['decision']}**. Development evidence only.",
            "",
            "Primary control: multiscale versus global-pool on the more-objects split.",
            "",
            "```json",
            json.dumps(result["comparisons"], indent=2, allow_nan=False),
            "```",
            "",
            result["claim_boundary"],
        ]
        (args.output / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        tracker.transition("complete")
        write_manifest(args.output)
    except BaseException as exc:
        if tracker.payload["state"] != "failed":
            tracker.transition("failed", error=exc)
        write_json(
            args.output / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "message": str(exc)},
        )
        raise
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
