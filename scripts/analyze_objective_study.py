from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ripii.utils.statistics import paired_summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_output(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("objective analysis must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = _signature(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("objective analysis is invalid JSON") from exc
    comparisons = payload.get("comparisons") if isinstance(payload, dict) else None
    if (
        payload.get("format") != "ripii-objective-study-analysis-v1"
        or payload.get("signature") != expected
        or not isinstance(comparisons, dict)
        or not comparisons
    ):
        raise ValueError("objective analysis signature or schema is invalid")
    supported = payload.get("supported_additions")
    expected_supported = []
    for mode, comparison in comparisons.items():
        reductions = comparison.get("relative_reduction_by_seed")
        passed = comparison.get("all_seed_five_percent_rule")
        if (
            not isinstance(reductions, list)
            or not reductions
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in reductions
            )
            or not isinstance(passed, bool)
            or passed != all(value >= 0.05 for value in reductions)
        ):
            raise ValueError("objective analysis contains an invalid comparison")
        if passed:
            expected_supported.append(mode)
    if supported != expected_supported or payload.get("decision") != (
        "one_or_more_additions_passed" if expected_supported else "no_auxiliary_advance"
    ):
        raise ValueError("objective analysis decision is inconsistent")
    return {
        "status": "PASS",
        "signature": payload["signature"],
        "comparisons_verified": len(comparisons),
    }


def analyze(summary_path: Path, protocol_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("summary is not bound to the supplied protocol")
    rows = summary.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("objective summary has no raw runs")
    grouped: dict[str, dict[int, dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid objective row")
        mode, seed, recon = row.get("mode"), row.get("seed"), row.get("recon")
        if (
            not isinstance(mode, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(recon, (int, float))
            or isinstance(recon, bool)
            or not math.isfinite(recon)
            or recon <= 0
            or seed in grouped.setdefault(mode, {})
        ):
            raise ValueError("invalid or duplicate objective cell")
        grouped[mode][seed] = row
    reference = grouped.get("simple_objective")
    if reference is None or len(reference) < 2:
        raise ValueError("simple objective reference is incomplete")
    seeds = sorted(reference)
    if any(sorted(cells) != seeds for cells in grouped.values()):
        raise ValueError("objective matrix is unpaired or incomplete")

    comparisons = {}
    supported = []
    for index, (mode, cells) in enumerate(sorted(grouped.items())):
        if mode == "simple_objective":
            continue
        candidate = [float(cells[seed]["recon"]) for seed in seeds]
        baseline = [float(reference[seed]["recon"]) for seed in seeds]
        reductions = [
            (baseline_value - candidate_value) / baseline_value
            for candidate_value, baseline_value in zip(
                candidate, baseline, strict=False
            )
        ]
        passes = all(value >= 0.05 for value in reductions)
        if passes:
            supported.append(mode)
        comparisons[mode] = {
            "candidate_reconstruction": candidate,
            "simple_reconstruction": baseline,
            "relative_reduction_by_seed": reductions,
            "all_seed_five_percent_rule": passes,
            "paired_statistics_candidate_minus_simple": paired_summary(
                candidate, baseline, bootstrap_seed=91_003 + index
            ),
        }
    result = {
        "format": "ripii-objective-study-analysis-v1",
        "evidence_status": "frozen_local_development",
        "input": {
            "summary": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "protocol": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
        },
        "source_sha256": {
            "scripts/analyze_objective_study.py": _sha256(Path(__file__)),
            "ripii/utils/statistics.py": _sha256(
                Path(__file__).resolve().parents[1] / "ripii/utils/statistics.py"
            ),
        },
        "experimental_unit": "paired initialization/minibatch seed",
        "seeds": seeds,
        "reference": "simple_objective",
        "effect_threshold": 0.05,
        "comparisons": comparisons,
        "supported_additions": supported,
        "decision": (
            "one_or_more_additions_passed" if supported else "no_auxiliary_advance"
        ),
        "interpretation": (
            "Prefer reconstruction+KL for this fixed-budget synthetic setting; "
            "three seeds do not establish population noninferiority or external utility."
        ),
    }
    result["signature"] = _signature(result)
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# Objective study v1 analysis",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "| comparison | mean candidate − simple MSE | seed reductions | passes |",
        "|---|---:|---|---|",
    ]
    for mode, comparison in result["comparisons"].items():
        difference = comparison["paired_statistics_candidate_minus_simple"][
            "mean_difference"
        ]
        reductions = ", ".join(
            f"{100 * value:.2f}%" for value in comparison["relative_reduction_by_seed"]
        )
        lines.append(
            f"| `{mode}` | {difference:.6f} | {reductions} | "
            f"{comparison['all_seed_five_percent_rule']} |"
        )
    lines.extend(
        [
            "",
            result["interpretation"],
            "",
            "Positive reduction means the addition helped; negative means it hurt.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the legacy objective study")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        if any(
            value is not None
            for value in (args.summary, args.protocol, args.output, args.markdown)
        ):
            parser.error("--verify-output cannot be combined with analysis arguments")
        print(json.dumps(verify_output(args.verify_output), indent=2))
        return
    if None in (args.summary, args.protocol, args.output, args.markdown):
        parser.error("analysis requires summary, protocol, JSON output, and Markdown")
    if args.output.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite objective analysis")
    result = analyze(args.summary, args.protocol)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(verify_output(args.output), indent=2))


if __name__ == "__main__":
    main()
