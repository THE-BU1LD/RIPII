from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ripii.utils.statistics import holm_adjust, paired_summary
from ripii.world.experiment import verify_capsule


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular non-symlink JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("input is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("input JSON root must be an object")
    provenance = {"name": path.name, "sha256": _sha256(path)}
    if payload.get("format") == "ripii-world-result-capsule-v2":
        verification = verify_capsule(path)
        provenance["capsule_verification"] = verification
        payload = json.loads(payload["retained"]["summary.json"]["content_text"])
    elif "format" in payload:
        raise ValueError("input must be a world summary or verified v2 result capsule")
    if not isinstance(payload.get("runs"), list):
        raise ValueError("input must be a world summary or v2 result capsule")
    return payload, provenance


def _metric(row: dict, split: str) -> float:
    try:
        value = float(row["metrics"][split]["position_rmse"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid position_rmse for split {split!r}") from exc
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite position_rmse for split {split!r}")
    return value


def analyze(
    input_path: Path,
    *,
    candidate: str,
    baseline: str,
    bottleneck: str,
) -> dict:
    if not candidate or not baseline or not bottleneck:
        raise ValueError("candidate, baseline, and bottleneck must be nonempty")
    if candidate == baseline:
        raise ValueError("candidate and baseline must be different variants")
    data, provenance = _load(input_path)
    splits = ("test", "more_objects", "composition", "fast")
    selected: dict[tuple[str, int], dict] = {}
    for row in data["runs"]:
        if not isinstance(row, dict):
            raise ValueError("each run must be an object")
        if row.get("bottleneck") != bottleneck or row.get("variant") not in {
            candidate,
            baseline,
        }:
            continue
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("selected run seeds must be nonnegative integers")
        key = (row["variant"], seed)
        if key in selected:
            raise ValueError(f"duplicate selected run for variant/seed {key!r}")
        selected[key] = row
    candidate_seeds = {seed for variant, seed in selected if variant == candidate}
    baseline_seeds = {seed for variant, seed in selected if variant == baseline}
    if not candidate_seeds or not baseline_seeds:
        raise ValueError("candidate and baseline must each contain at least one seed")
    if candidate_seeds != baseline_seeds:
        missing_candidate = sorted(baseline_seeds - candidate_seeds)
        missing_baseline = sorted(candidate_seeds - baseline_seeds)
        raise ValueError(
            "paired seed sets differ: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    seeds = sorted(candidate_seeds)
    analyses = {}
    for offset, split in enumerate(splits):
        candidate_values = [
            _metric(selected[(candidate, seed)], split) for seed in seeds
        ]
        baseline_values = [
            _metric(selected[(baseline, seed)], split) for seed in seeds
        ]
        analyses[split] = paired_summary(
            candidate_values, baseline_values, bootstrap_seed=10_000 + offset
        )
    adjusted = holm_adjust(
        [analyses[split]["exact_two_sided_sign_flip_p"] for split in splits]
    )
    for split, value in zip(splits, adjusted):
        analyses[split]["holm_adjusted_p"] = value
    root = Path(__file__).resolve().parents[1]
    return {
        "evidence_status": "development_evidence",
        "experimental_unit": "initialization/minibatch seed on one fixed synthetic dataset",
        "input": provenance,
        "source_sha256": {
            "scripts/analyze_world.py": _sha256(Path(__file__)),
            "ripii/utils/statistics.py": _sha256(root / "ripii/utils/statistics.py"),
        },
        "candidate": candidate,
        "baseline": baseline,
        "bottleneck": bottleneck,
        "seeds": seeds,
        "metrics": analyses,
        "interpretation_limit": "Exact tests have very low resolution at this seed count; bootstrap intervals describe these seeds and are not population confidence intervals.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired seed-level world-model statistics"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--candidate", default="multiscale")
    parser.add_argument("--baseline", default="graph")
    parser.add_argument("--bottleneck", default="continuous")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.input,
        candidate=args.candidate,
        baseline=args.baseline,
        bottleneck=args.bottleneck,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
