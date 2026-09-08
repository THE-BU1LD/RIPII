from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ripii.utils.power import approximate_paired_seed_count
from ripii.world.experiment import verify_capsule, write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_plan_artifact(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("power plan must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = _signature(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("power plan is invalid or non-finite JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "ripii-paired-power-plan-v1"
        or payload.get("signature") != expected
        or payload.get("recommended_minimum_pairs", 0) < 6
        or payload.get("minimum_detectable_effect", 0) <= 0
        or payload.get("evidence_status")
        != "prospective_planning_from_development_variance"
    ):
        raise ValueError("power-plan signature or schema is invalid")
    return {
        "status": "PASS",
        "signature": payload["signature"],
        "recommended_minimum_pairs": payload["recommended_minimum_pairs"],
    }


def _load_summary(path: Path) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("planning input must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = {"name": path.name, "sha256": _sha256(path)}
    if payload.get("format") == "ripii-world-result-capsule-v2":
        provenance["capsule_verification"] = verify_capsule(path)
        payload = json.loads(payload["retained"]["summary.json"]["content_text"])
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("planning input must contain world benchmark runs")
    return payload, provenance


def plan(
    path: Path,
    *,
    candidate: str,
    baseline: str,
    split: str,
    bottleneck: str,
    minimum_detectable_effect: float,
    alpha: float,
    power: float,
) -> dict:
    summary, provenance = _load_summary(path)
    selected = {}
    for row in summary["runs"]:
        if (
            isinstance(row, dict)
            and row.get("variant") in {candidate, baseline}
            and row.get("bottleneck") == bottleneck
        ):
            seed = row.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("selected rows require nonnegative integer seeds")
            key = (row["variant"], seed)
            if key in selected:
                raise ValueError("duplicate candidate/baseline seed")
            selected[key] = row
    candidate_seeds = {seed for variant, seed in selected if variant == candidate}
    baseline_seeds = {seed for variant, seed in selected if variant == baseline}
    if not candidate_seeds or candidate_seeds != baseline_seeds:
        raise ValueError("candidate and baseline require identical paired seeds")
    seeds = sorted(candidate_seeds)
    differences = []
    for seed in seeds:
        try:
            left = float(selected[(candidate, seed)]["metrics"][split]["position_rmse"])
            right = float(selected[(baseline, seed)]["metrics"][split]["position_rmse"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid planning metric") from exc
        if right <= 0:
            raise ValueError("planning baseline metric must be positive")
        differences.append(left / right - 1.0)
    result = approximate_paired_seed_count(
        differences,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
    )
    root = Path(__file__).resolve().parents[1]
    result.update(
        {
            "evidence_status": "prospective_planning_from_development_variance",
            "input": provenance,
            "candidate": candidate,
            "baseline": baseline,
            "bottleneck": bottleneck,
            "metric": f"{split} position_rmse relative difference",
            "seeds": seeds,
            "source_sha256": {
                "scripts/plan_power.py": _sha256(Path(__file__)),
                "ripii/utils/power.py": _sha256(root / "ripii/utils/power.py"),
            },
        }
    )
    result["signature"] = _signature(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan future paired RIPII seed count")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--candidate", default="multiscale")
    parser.add_argument("--baseline", default="graph")
    parser.add_argument("--split", default="more_objects")
    parser.add_argument("--bottleneck", default="continuous")
    parser.add_argument("--minimum-detectable-effect", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        if args.input is not None or args.output is not None:
            parser.error("--verify-output cannot be combined with planning arguments")
        print(json.dumps(verify_plan_artifact(args.verify_output), indent=2))
        return
    if args.input is None or args.output is None:
        parser.error("planning requires input and --output")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing power-plan output: {args.output}")
    result = plan(
        args.input,
        candidate=args.candidate,
        baseline=args.baseline,
        split=args.split,
        bottleneck=args.bottleneck,
        minimum_detectable_effect=args.minimum_detectable_effect,
        alpha=args.alpha,
        power=args.power,
    )
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
