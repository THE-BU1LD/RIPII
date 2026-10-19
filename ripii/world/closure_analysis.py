"""Pure paired-unit analysis and figures for the frozen closure study.

This module reads no repository files at import or analysis time.  The caller
supplies the complete protocol and retained phase measurements.  It never treats
particles, timesteps, or evaluation regimes as independent replications.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("test", "more_objects", "composition", "fast")
OOD_SPLITS = SPLITS[1:]
PROFILE_FIELDS = tuple(f"seconds_per_forward_n{n}" for n in (2, 4, 8))
_TOL = 1e-12


def _number(value: Any, *, nonnegative: bool = True) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        return None
    return result


def _mean(values: list[float]) -> float:
    # Divide before summing so finite large observations do not overflow a sum.
    return math.fsum(value / len(values) for value in values)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": _mean(values),
        "min": min(values),
        "max": max(values),
    }


def _metric_values(
    values: Any, location: str, errors: list[str], prefix: str = ""
) -> dict[str, float]:
    """Keep scalar metrics; flatten nested metric dictionaries, never samples."""
    if not isinstance(values, dict):
        errors.append(f"{location}: expected a metric object")
        return {}
    result: dict[str, float] = {}
    for name, value in values.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            result.update(_metric_values(value, location, errors, f"{key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = _number(value, nonnegative="rmse" in key.lower())
            if number is None:
                errors.append(f"{location}.{key}: invalid scalar metric")
            else:
                result[key] = number
    return result


def _protocol(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[tuple[int, int]]]:
    if payload.get("format") != "ripii-closure-phase-results-v1":
        raise ValueError("unsupported closure phase-results format")
    phase = payload.get("phase")
    if phase not in ("dry_run", "development", "confirmation"):
        raise ValueError("phase must be dry_run, development, or confirmation")
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("format") != "ripii-closure-study-protocol-v1"
    ):
        raise ValueError("missing or unsupported closure protocol")
    if protocol.get("dataset", {}).get("evaluation_splits") != list(SPLITS):
        raise ValueError("closure v1 requires the four prespecified evaluation splits")
    statistics = protocol.get("statistics", {})
    required = {
        "primary_candidate": "hier_env_s4",
        "primary_control": "global_compute",
        "practical_ratio": 0.95,
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 46001,
        "interval_percentiles": [2.5, 97.5],
        "alpha": 0.05,
        "iid_max_ratio": 1.05,
        "required_other_controls": ["mean_env_s4", "law_env_s4"],
    }
    if any(statistics.get(key) != value for key, value in required.items()):
        raise ValueError("statistical constants do not match closure study v1")
    arm_specs = protocol.get("arms")
    if not isinstance(arm_specs, list) or not all(
        isinstance(arm, dict) for arm in arm_specs
    ):
        raise ValueError("protocol arms must be a list of objects")
    arms = [arm.get("id") for arm in arm_specs]
    compute_id = protocol.get("compute_control", {}).get("id")
    if compute_id != "global_compute" or not all(isinstance(arm, str) for arm in arms):
        raise ValueError("invalid arm identifiers or compute control")
    arms.append(compute_id)
    if len(set(arms)) != len(arms) or not {
        "hier_env_s4",
        "global_env_s4",
        "global_compute",
        "mean_env_s4",
        "law_env_s4",
        "oracle",
    }.issubset(arms):
        raise ValueError("duplicate or missing required protocol arms")
    seeds = protocol.get("seeds", {}).get(phase)
    if phase == "dry_run":
        seeds = [seeds]
    required_n = {"dry_run": 1, "development": 3, "confirmation": 6}[phase]
    if not isinstance(seeds, list) or len(seeds) != required_n:
        raise ValueError(f"{phase} requires exactly {required_n} protocol seed pairs")
    expected: list[tuple[int, int]] = []
    for seed in seeds:
        if not isinstance(seed, dict) or any(
            type(seed.get(key)) is not int for key in ("data", "model")
        ):
            raise ValueError("protocol seeds must be integer data/model pairs")
        expected.append((seed["data"], seed["model"]))
    if len(set(expected)) != len(expected) or len(
        {pair[0] for pair in expected}
    ) != len(expected):
        raise ValueError(
            "independent protocol units must have distinct data seeds and seed pairs"
        )
    return protocol, arms, expected


def _collect(
    payload: dict[str, Any], arms: list[str], expected: list[tuple[int, int]]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    errors: list[str] = []
    missing: list[str] = []
    raw_units = payload.get("units")
    if not isinstance(raw_units, list):
        raise ValueError("units must be a list")
    if type(payload.get("complete")) is not bool:
        raise ValueError("complete must be an explicit boolean")
    if not payload["complete"]:
        missing.append("phase is recorded as incomplete")
    units: list[dict[str, Any]] = []
    seen: list[tuple[int, int]] = []
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict) or any(
            type(raw.get(key)) is not int for key in ("data_seed", "model_seed")
        ):
            errors.append(f"unit[{index}]: missing integer seed identity")
            continue
        pair = (raw["data_seed"], raw["model_seed"])
        label = f"unit {pair[0]}/{pair[1]}"
        if pair not in expected:
            errors.append(f"{label}: unexpected seed pair")
        if pair in seen:
            errors.append(f"{label}: duplicate independent unit")
        seen.append(pair)
        raw_arms = raw.get("arms", {})
        if not isinstance(raw_arms, dict):
            errors.append(f"{label}: arms must be an object")
            raw_arms = {}
        if set(raw_arms) - set(arms):
            errors.append(
                f"{label}: unexpected arms {sorted(set(raw_arms) - set(arms))}"
            )
        unit: dict[str, Any] = {"data_seed": pair[0], "model_seed": pair[1], "arms": {}}
        for arm in arms:
            raw_arm = raw_arms.get(arm)
            location = f"{label}/{arm}"
            if not isinstance(raw_arm, dict):
                missing.append(f"{location}: missing arm")
                continue
            record: dict[str, Any] = {"metrics": {}, "profile": {}, "noise_metrics": {}}
            for field in ("parameters", "completed_updates", "optimization_seconds"):
                number = _number(raw_arm.get(field))
                if field not in raw_arm:
                    missing.append(f"{location}: missing {field}")
                elif number is None or (
                    field != "optimization_seconds" and type(raw_arm[field]) is not int
                ):
                    errors.append(f"{location}: invalid {field}")
                else:
                    record[field] = raw_arm[field]
            record["compute_coverage_ratio"] = None
            coverage = raw_arm.get("compute_coverage_ratio")
            if coverage is not None:
                valid_coverage = _number(coverage)
                if valid_coverage is None:
                    errors.append(f"{location}: invalid compute_coverage_ratio")
                else:
                    record["compute_coverage_ratio"] = valid_coverage
            raw_metrics = raw_arm.get("metrics", {})
            if not isinstance(raw_metrics, dict):
                errors.append(f"{location}: metrics must be an object")
                raw_metrics = {}
            for split in SPLITS:
                if split not in raw_metrics:
                    missing.append(f"{location}: missing split {split}")
                    continue
                values = _metric_values(
                    raw_metrics[split], f"{location}/{split}", errors
                )
                for metric in ("position_rmse", "velocity_rmse"):
                    if metric not in values:
                        missing.append(f"{location}/{split}: missing valid {metric}")
                record["metrics"][split] = values
            if "noise_metrics" in raw_arm:
                record["noise_metrics"] = _metric_values(
                    raw_arm["noise_metrics"], f"{location}/noise", errors
                )
            raw_profile = raw_arm.get("profile", {})
            if isinstance(raw_profile, dict):
                for field in PROFILE_FIELDS:
                    if field in raw_profile:
                        value = _number(raw_profile[field])
                        if value is None:
                            errors.append(f"{location}/profile: invalid {field}")
                        else:
                            record["profile"][field] = value
            else:
                errors.append(f"{location}: profile must be an object")
            ood = [
                record["metrics"].get(split, {}).get("position_rmse")
                for split in OOD_SPLITS
            ]
            record["ood_position_rmse"] = (
                _mean(ood) if all(value is not None for value in ood) else None
            )
            unit["arms"][arm] = record
        units.append(unit)
    for pair in expected:
        if pair not in seen:
            missing.append(f"missing independent unit {pair[0]}/{pair[1]}")
    order = {pair: index for index, pair in enumerate(expected)}
    units.sort(
        key=lambda unit: order.get(
            (unit["data_seed"], unit["model_seed"]), len(expected)
        )
    )
    return units, errors, missing


def _aggregate(units: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for arm in arms:
        rows = [unit["arms"][arm] for unit in units if arm in unit["arms"]]
        summary: dict[str, Any] = {"n_units": len(rows), "regimes": {}}
        for field in (
            "parameters",
            "completed_updates",
            "optimization_seconds",
            "ood_position_rmse",
            "compute_coverage_ratio",
        ):
            summary[field] = _summary(
                [float(row[field]) for row in rows if row.get(field) is not None]
            )
        for split in SPLITS:
            names = sorted(
                {name for row in rows for name in row["metrics"].get(split, {})}
            )
            summary["regimes"][split] = {
                name: _summary(
                    [
                        row["metrics"][split][name]
                        for row in rows
                        if name in row["metrics"].get(split, {})
                    ]
                )
                for name in names
            }
        for group in ("noise_metrics", "profile"):
            names = sorted({name for row in rows for name in row[group]})
            summary[group] = {
                name: _summary([row[group][name] for row in rows if name in row[group]])
                for name in names
            }
        summaries[arm] = summary
    return summaries


def _sign_flip_p(log_ratios: list[float], null_log: float) -> float:
    differences = np.asarray(log_ratios, dtype=np.float64) - null_log
    observed = abs(float(np.mean(differences)))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    statistics = np.abs(np.mean(signs * differences, axis=1))
    tolerance = _TOL * max(1.0, observed)
    return float(np.count_nonzero(statistics >= observed - tolerance) / len(statistics))


def _below(value: float | None, threshold: float) -> bool:
    return (
        value is not None
        and value < threshold
        and not math.isclose(value, threshold, rel_tol=_TOL, abs_tol=_TOL)
    )


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and (
        value >= threshold or math.isclose(value, threshold, rel_tol=_TOL, abs_tol=_TOL)
    )


def _paired(
    units: list[dict[str, Any]],
    candidate: str,
    control: str,
    statistics: dict[str, Any],
    *,
    iid: bool = False,
) -> dict[str, Any]:
    effects: list[dict[str, Any]] = []
    logs: list[float] = []
    unavailable: list[str] = []
    for unit in units:
        label = f"{unit['data_seed']}/{unit['model_seed']}"
        values: list[float | None] = []
        for arm in (candidate, control):
            row = unit["arms"].get(arm, {})
            values.append(
                row.get("metrics", {}).get("test", {}).get("position_rmse")
                if iid
                else row.get("ood_position_rmse")
            )
        left, right = values
        effect: dict[str, Any] = {
            "data_seed": unit["data_seed"],
            "model_seed": unit["model_seed"],
            "candidate_rmse": left,
            "control_rmse": right,
            "ratio": None,
            "log_ratio": None,
        }
        if left is None or right is None:
            unavailable.append(f"{label}: missing paired metric")
        elif left <= 0 or right <= 0:
            unavailable.append(
                f"{label}: zero error makes the prespecified log ratio unavailable; no pseudocount used"
            )
        else:
            ratio = left / right
            if not math.isfinite(ratio) or ratio <= 0:
                unavailable.append(
                    f"{label}: ratio outside finite positive floating-point range"
                )
            else:
                log_ratio = math.log(ratio)
                effect.update(ratio=ratio, log_ratio=log_ratio)
                logs.append(log_ratio)
        effects.append(effect)
    result: dict[str, Any] = {
        "candidate": candidate,
        "control": control,
        "metric": "iid_position_rmse" if iid else "equal_mean_ood_position_rmse",
        "n_pairs": len(logs),
        "unit_effects": effects,
        "unavailable": unavailable,
        "geometric_ratio": None,
        "interval_95": None,
        "primary_sign_flip_p": None,
        "practical_evidence": "UNAVAILABLE",
        "interval_scope": "paired independent-unit percentile bootstrap; descriptive, especially with six units",
    }
    # Never quietly bootstrap a selected positive-error subset.
    if not logs or unavailable:
        return result
    mean_log = _mean(logs)
    rng = np.random.default_rng(statistics["bootstrap_seed"])
    indices = rng.integers(
        0, len(logs), size=(statistics["bootstrap_repetitions"], len(logs))
    )
    bootstrap = np.exp(np.mean(np.asarray(logs)[indices], axis=1))
    low, high = np.percentile(bootstrap, statistics["interval_percentiles"])
    ratio = math.exp(mean_log)
    if not all(
        math.isfinite(value) and value > 0 for value in (ratio, float(low), float(high))
    ):
        result["unavailable"].append(
            "geometric summary outside finite positive floating-point range"
        )
        return result
    threshold = statistics["iid_max_ratio"] if iid else statistics["practical_ratio"]
    result.update(geometric_ratio=ratio, interval_95=[float(low), float(high)])
    if _below(float(high), threshold):
        result["practical_evidence"] = "INTERVAL_BELOW_THRESHOLD"
    elif _at_least(float(low), threshold):
        result["practical_evidence"] = "INTERVAL_AT_OR_ABOVE_THRESHOLD"
    else:
        result["practical_evidence"] = "INTERVAL_CROSSES_THRESHOLD"
    return result


def analyze_phase(payload: dict[str, Any]) -> dict[str, Any]:
    """Analyze retained observations; incomplete or invalid evidence cannot win.

    Unsupported schema/protocol constants raise ``ValueError``.  Measurement
    defects are retained as an INVALID analysis, while incomplete grids, zero
    log-ratio inputs and insufficient compute coverage remain INCONCLUSIVE.
    Development and dry-run outcomes never receive a confirmatory H2 decision.
    """
    protocol, arms, expected = _protocol(payload)
    units, errors, missing = _collect(payload, arms, expected)
    statistics = protocol["statistics"]
    candidate = statistics["primary_candidate"]
    primary_control = statistics["primary_control"]
    controls = [primary_control, *statistics["required_other_controls"]]
    comparisons = {
        control: _paired(units, candidate, control, statistics) for control in controls
    }
    iid = _paired(units, candidate, primary_control, statistics, iid=True)
    coverage_rows: list[dict[str, Any]] = []
    for unit in units:
        hierarchy = unit["arms"].get(candidate, {})
        control = unit["arms"].get(primary_control, {})
        reported = control.get("compute_coverage_ratio")
        ht, ct = (
            hierarchy.get("optimization_seconds"),
            control.get("optimization_seconds"),
        )
        measured = ct / ht if ht is not None and ht > 0 and ct is not None else None
        if measured is not None and not math.isfinite(measured):
            errors.append("nonfinite compute coverage derived from optimization times")
            measured = None
        if (
            reported is not None
            and measured is not None
            and not math.isclose(reported, measured, rel_tol=1e-6, abs_tol=1e-9)
        ):
            errors.append(
                f"unit {unit['data_seed']}/{unit['model_seed']}: reported compute coverage disagrees with optimization times"
            )
        coverage_rows.append(
            {
                "data_seed": unit["data_seed"],
                "model_seed": unit["model_seed"],
                "reported_ratio": reported,
                "measured_ratio": measured,
                "covered": reported is not None
                and measured is not None
                and reported >= 1.0
                and measured >= 1.0,
            }
        )
        for flat in ("global_env_s4", primary_control):
            hp = hierarchy.get("parameters")
            cp = unit["arms"].get(flat, {}).get("parameters")
            if hp is not None and cp is not None and hp != cp:
                errors.append(
                    f"unit {unit['data_seed']}/{unit['model_seed']}: {flat} is not parameter-matched to hierarchy"
                )
    complete = not errors and not missing and len(units) == len(expected)
    compute_complete = len(coverage_rows) == len(expected) and all(
        row["covered"] for row in coverage_rows
    )
    primary = comparisons[primary_control]
    if (
        payload["phase"] == "confirmation"
        and complete
        and not primary["unavailable"]
        and primary["n_pairs"] == 6
    ):
        primary["primary_sign_flip_p"] = _sign_flip_p(
            [row["log_ratio"] for row in primary["unit_effects"]],
            math.log(statistics["practical_ratio"]),
        )
    interval = primary["interval_95"]
    iid_ratio = iid["geometric_ratio"]
    safeguards = {
        "complete_grid": complete,
        "six_confirmation_units": payload["phase"] == "confirmation"
        and len(units) == 6,
        "compute_coverage_complete": compute_complete,
        "primary_point_below_095": _below(
            primary["geometric_ratio"], statistics["practical_ratio"]
        ),
        "primary_upper_below_095": interval is not None
        and _below(interval[1], statistics["practical_ratio"]),
        "primary_p_at_most_005": primary["primary_sign_flip_p"] is not None
        and primary["primary_sign_flip_p"] <= statistics["alpha"],
        "iid_point_at_most_105": iid_ratio is not None
        and not _below(statistics["iid_max_ratio"], iid_ratio),
        "simpler_control_points_below_095": all(
            _below(
                comparisons[control]["geometric_ratio"], statistics["practical_ratio"]
            )
            for control in statistics["required_other_controls"]
        ),
    }
    no_benefit = [
        control
        for control, comparison in comparisons.items()
        if comparison["interval_95"] is not None
        and _at_least(comparison["interval_95"][0], statistics["practical_ratio"])
    ]
    unavailable = any(
        comparison["unavailable"] or comparison["geometric_ratio"] is None
        for comparison in [*comparisons.values(), iid]
    )
    if errors:
        decision, reason = (
            "INVALID",
            "Measurement/schema or parameter-matching defects prevent a scientific classification.",
        )
    elif payload["phase"] != "confirmation":
        decision, reason = (
            "NOT_TESTED",
            "Dry-run and development results are descriptive; no confirmatory H2 test or decision is made.",
        )
    elif not complete:
        decision, reason = (
            "INCONCLUSIVE",
            "The prespecified six-unit grid is incomplete.",
        )
    elif not compute_complete:
        decision, reason = (
            "INCONCLUSIVE",
            "The flat control did not verifiably cover the hierarchy's optimization time in every unit.",
        )
    elif unavailable:
        decision, reason = (
            "INCONCLUSIVE",
            "At least one required prespecified log comparison is unavailable; no pseudocount or unit exclusion is permitted.",
        )
    elif all(safeguards.values()):
        decision, reason = (
            "SUPPORTED",
            "All frozen primary, IID, simpler-control and compute-coverage criteria are satisfied.",
        )
    elif no_benefit:
        decision, reason = (
            "NEGATIVE",
            "At least one required control's lower descriptive ratio bound is at or above 0.95, triggering the frozen negative rule.",
        )
    else:
        decision, reason = (
            "INCONCLUSIVE",
            "The complete evidence satisfies neither the frozen support rule nor its negative rule.",
        )
    per_regime_direction = []
    for split in SPLITS:
        pairs = [
            (
                unit["arms"]
                .get(candidate, {})
                .get("metrics", {})
                .get(split, {})
                .get("position_rmse"),
                unit["arms"]
                .get(primary_control, {})
                .get("metrics", {})
                .get(split, {})
                .get("position_rmse"),
            )
            for unit in units
        ]
        if pairs and all(
            left is not None and right is not None for left, right in pairs
        ):
            per_regime_direction.append(
                _mean([left for left, _ in pairs])
                < _mean([right for _, right in pairs])
            )
    mixed = (
        bool(per_regime_direction)
        and any(per_regime_direction)
        and not all(per_regime_direction)
    )
    return {
        "format": "ripii-closure-analysis-v1",
        "study_id": protocol.get("study_id"),
        "phase": payload["phase"],
        "complete": complete,
        "expected_units": len(expected),
        "observed_units": len(units),
        "validation": {"errors": errors, "missing": missing},
        "statistics": statistics,
        "arm_order": arms,
        "split_order": list(SPLITS),
        "unit_outcomes": units,
        "arm_summaries": _aggregate(units, arms),
        "comparisons": comparisons,
        "iid_comparison": iid,
        "compute_coverage": {"complete": compute_complete, "units": coverage_rows},
        "decision_h2": decision,
        "h2": {
            "decision": decision,
            "reason": reason,
            "safeguards": safeguards,
            "controls_at_or_above_practical_bound": no_benefit,
            "descriptive_mixed_regime_direction": mixed,
        },
        "caveats": [
            "Independent units are paired data/model-seed draws, not particles, timesteps, or regimes.",
            "Each unit's OOD endpoint is the equal arithmetic mean of three regime RMSEs; paired geometric ratios are computed after that averaging.",
            "The 10000-draw paired-unit percentile bootstrap uses seed 46001. With six units its 95% interval is descriptive and may have poor coverage.",
            "The only significance test is the two-sided exhaustive sign flip of primary log ratios centered at log(0.95); symmetry/exchangeability is assumed, not guaranteed by randomized treatment.",
            "Secondary comparisons, noise, horizon, count, and profile results are descriptive. Development and dry-run p-values are not computed.",
            "A 1e-12 relative/absolute comparison tolerance handles floating-point threshold ties; exact-threshold results cannot pass a strict support criterion.",
            "Zero oracle error is valid and excluded from learned-winner log tests. No error pseudocount is added to numerical results.",
            "Optimization time and small-fixture inference timings describe this execution; neither proves a general causal efficiency advantage or large-N scaling.",
            "This diagnostic establishes neither architectural novelty nor SOTA superiority; historical negative studies remain separate evidence.",
        ],
    }


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{value:.6g}"


def _markdown(analysis: dict[str, Any]) -> str:
    phase = analysis["phase"]
    lines = [
        f"# Closure study v1 — {phase} results",
        "",
        f"**H2: {analysis['decision_h2']}**. {analysis['h2']['reason']}",
        "",
        f"Retained independent units: {analysis['observed_units']}/{analysis['expected_units']}. Complete validated grid: {analysis['complete']}. Compute coverage: {analysis['compute_coverage']['complete']}.",
        "",
        "H1 correctness is classified by the separate mathematical/parity checks, not by these learned-model comparisons.",
        "",
        "## Paired hierarchy comparisons",
        "",
        "Candidate is `hier_env_s4`. Ratios below one favor the hierarchy. OOD is an equal mean of regime RMSEs within each independent unit.",
        "",
        "| Control | Pairs | Geometric OOD ratio | Bootstrap 95% interval | Primary sign-flip p | Interval evidence |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for control, result in analysis["comparisons"].items():
        interval = result["interval_95"]
        bounds = (
            "NA" if interval is None else f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
        )
        lines.append(
            f"| {control} | {result['n_pairs']} | {_fmt(result['geometric_ratio'])} | {bounds} | {_fmt(result['primary_sign_flip_p'])} | {result['practical_evidence']} |"
        )
    iid = analysis["iid_comparison"]
    lines += [
        "",
        f"IID hierarchy/global_compute geometric ratio: **{_fmt(iid['geometric_ratio'])}**; maximum permitted point ratio is 1.05.",
        "",
        "### Frozen decision safeguards",
        "",
        "| Safeguard | Satisfied |",
        "| --- | --- |",
    ]
    lines += [
        f"| {name} | {value} |" for name, value in analysis["h2"]["safeguards"].items()
    ]
    lines += [
        "",
        "## Arithmetic regime means",
        "",
        "These are arithmetic means over independent units, not geometric ratios or pooled particle errors.",
        "",
        "| Arm | IID position | More objects | Composition | Fast | Mean OOD | Optimization seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in analysis["arm_order"]:
        row = analysis["arm_summaries"][arm]
        values = [
            _fmt(row["regimes"][split].get("position_rmse", {}).get("mean"))
            for split in SPLITS
        ]
        lines.append(
            f"| {arm} | "
            + " | ".join(
                values
                + [
                    _fmt(row["ood_position_rmse"]["mean"]),
                    _fmt(row["optimization_seconds"]["mean"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Every independent unit and arm",
        "",
        "Full per-regime metrics, optional horizon/noise metrics and measured profiles are retained in `analysis.json` and `tables.csv`.",
        "",
        "| Data seed | Model seed | Arm | OOD position RMSE | IID position RMSE | Optimization seconds | Updates | Compute ratio |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for unit in analysis["unit_outcomes"]:
        for arm in analysis["arm_order"]:
            if arm not in unit["arms"]:
                continue
            row = unit["arms"][arm]
            values = [
                row.get("ood_position_rmse"),
                row["metrics"].get("test", {}).get("position_rmse"),
                row.get("optimization_seconds"),
                row.get("completed_updates"),
                row.get("compute_coverage_ratio"),
            ]
            lines.append(
                f"| {unit['data_seed']} | {unit['model_seed']} | {arm} | "
                + " | ".join(_fmt(value) for value in values)
                + " |"
            )
    lines += ["", "## Evidence limitations and failures", ""]
    messages = [*analysis["validation"]["errors"], *analysis["validation"]["missing"]]
    for comparison in [*analysis["comparisons"].values(), analysis["iid_comparison"]]:
        messages.extend(
            f"{comparison['control']} ({comparison['metric']}): {message}"
            for message in comparison["unavailable"]
        )
    lines += [f"- {message}" for message in messages] or [
        "No input measurement or grid errors were detected by this analysis. Source, split, prediction and checkpoint integrity require the separate artifact verification."
    ]
    lines += [
        "",
        "## Interpretation limits",
        "",
        *[f"- {text}" for text in analysis["caveats"]],
        "",
        "## Figures",
        "",
        "- `regime_heatmap`: arithmetic position RMSE by arm and regime; logarithmic colors apply only to positive values. Exact zeros and missing cells are explicitly marked.",
        "- `paired_effects`: every available paired OOD ratio, geometric ratio and descriptive bootstrap interval against the required controls.",
        "- `accuracy_time`: mean OOD position RMSE versus recorded mean optimization time. Exact zeros use a labeled plotting position only; numerical tables are unchanged.",
        "",
    ]
    return "\n".join(lines)


def _tables(analysis: dict[str, Any], path: Path) -> None:
    fields = [
        "scope",
        "phase",
        "data_seed",
        "model_seed",
        "arm",
        "regime",
        "metric",
        "value",
        "n",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for unit in analysis["unit_outcomes"]:
            for arm, row in unit["arms"].items():
                groups = {
                    **row["metrics"],
                    "noise": row["noise_metrics"],
                    "profile": row["profile"],
                    "run": {
                        name: row.get(name)
                        for name in (
                            "parameters",
                            "optimization_seconds",
                            "completed_updates",
                            "compute_coverage_ratio",
                            "ood_position_rmse",
                        )
                    },
                }
                for regime, metrics in groups.items():
                    for metric, value in metrics.items():
                        writer.writerow(
                            {
                                "scope": "unit",
                                "phase": analysis["phase"],
                                "data_seed": unit["data_seed"],
                                "model_seed": unit["model_seed"],
                                "arm": arm,
                                "regime": regime,
                                "metric": metric,
                                "value": value,
                                "n": 1 if value is not None else 0,
                            }
                        )
        for arm, row in analysis["arm_summaries"].items():
            groups = {
                **row["regimes"],
                "noise": row["noise_metrics"],
                "profile": row["profile"],
                "run": {
                    name: row[name]
                    for name in (
                        "parameters",
                        "optimization_seconds",
                        "completed_updates",
                        "compute_coverage_ratio",
                        "ood_position_rmse",
                    )
                },
            }
            for regime, metrics in groups.items():
                for metric, summary in metrics.items():
                    writer.writerow(
                        {
                            "scope": "arithmetic_mean",
                            "phase": analysis["phase"],
                            "arm": arm,
                            "regime": regime,
                            "metric": metric,
                            "value": summary["mean"],
                            "n": summary["n"],
                        }
                    )


def _figures(analysis: dict[str, Any], output: Path) -> None:
    # Object-oriented Agg avoids changing the caller's pyplot/backend state.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    def save(figure: Figure, stem: str) -> None:
        FigureCanvasAgg(figure)
        figure.savefig(
            output / f"{stem}.png",
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "RIPII closure analysis"},
        )
        figure.savefig(
            output / f"{stem}.pdf",
            bbox_inches="tight",
            metadata={
                "Creator": "RIPII closure analysis",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        figure.clear()

    arms = analysis["arm_order"]
    summaries = analysis["arm_summaries"]
    phase_label = (
        "Protected confirmation"
        if analysis["phase"] == "confirmation"
        else f"{analysis['phase'].replace('_', ' ').title()} — descriptive only"
    )
    data = np.asarray(
        [
            [
                summaries[arm]["regimes"][split]
                .get("position_rmse", {})
                .get("mean", np.nan)
                for split in SPLITS
            ]
            for arm in arms
        ],
        dtype=float,
    )
    positive = data[np.isfinite(data) & (data > 0)]
    vmin = float(positive.min()) if positive.size else 1e-6
    vmax = max(float(positive.max()), vmin * 1.01) if positive.size else 1.0
    figure = Figure(figsize=(8.8, max(4.8, 0.39 * len(arms) + 1.9)))
    axis = figure.subplots()
    masked = np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)
    mesh = axis.imshow(
        masked, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis_r", aspect="auto"
    )
    axis.set_xticks(range(len(SPLITS)), ["IID", "More objects", "Composition", "Fast"])
    axis.set_yticks(range(len(arms)), arms, fontsize=9)
    for i in range(len(arms)):
        for j in range(len(SPLITS)):
            value = data[i, j]
            label = (
                "NA"
                if not math.isfinite(value)
                else "0 (exact)"
                if value == 0
                else f"{value:.3g}"
            )
            color = (
                "white"
                if math.isfinite(value)
                and value > 0
                and float(np.asarray(mesh.norm(value)).item()) > 0.55
                else "black"
            )
            axis.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    axis.set_title(f"Position RMSE by regime\n{phase_label}", loc="left", pad=12)
    figure.colorbar(
        mesh, ax=axis, label="Position RMSE (positive values; log colors)", shrink=0.85
    )
    figure.text(
        0.05,
        0.02,
        "Arithmetic independent-unit means. Exact zero has no logarithmic color; NA means unavailable.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, "regime_heatmap")

    figure = Figure(figsize=(8.6, 4.6))
    axis = figure.subplots()
    comparisons = analysis["comparisons"]
    shown: list[float] = []
    for i, comparison in enumerate(comparisons.values()):
        points = [
            effect["ratio"]
            for effect in comparison["unit_effects"]
            if effect["ratio"] is not None
        ]
        if points:
            offsets = (
                np.linspace(-0.16, 0.16, len(points))
                if len(points) > 1
                else np.asarray([0.0])
            )
            axis.scatter(
                points,
                i + offsets,
                color="#2563eb",
                alpha=0.8,
                s=35,
                label="Independent units" if i == 0 else None,
                zorder=3,
            )
            shown.extend(points)
        center, interval = comparison["geometric_ratio"], comparison["interval_95"]
        if center is not None and interval is not None:
            summary_y = i - 0.29
            axis.plot(interval, [summary_y, summary_y], color="#111827", linewidth=2)
            axis.scatter(
                [center],
                [summary_y],
                color="#111827",
                marker="D",
                s=45,
                label="Geometric ratio and bootstrap interval" if i == 0 else None,
                zorder=4,
            )
            shown.extend(interval)
        if not points:
            axis.text(1.02, i, "Unavailable", va="center", fontsize=9)
    axis.axvline(
        0.95, color="#b45309", linestyle="--", label="Prespecified 0.95 threshold"
    )
    axis.axvline(1.0, color="#6b7280", linestyle=":", label="Equal error")
    axis.set_xscale("log")
    axis.set_yticks(range(len(comparisons)), list(comparisons))
    axis.set_ylim(-0.5, len(comparisons) - 0.5)
    if shown:
        axis.set_xlim(min([0.9, *shown]) / 1.15, max([1.05, *shown]) * 1.15)
    else:
        axis.set_xlim(0.8, 1.3)
    axis.set_xlabel("Hierarchy / control OOD RMSE ratio (log axis)")
    axis.set_title(f"Every paired OOD effect\n{phase_label}", loc="left")
    axis.grid(axis="x", alpha=0.18)
    axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    figure.tight_layout()
    save(figure, "paired_effects")

    figure = Figure(figsize=(10.0, 5.4))
    axis = figure.subplots()
    y_positive = [
        row["ood_position_rmse"]["mean"]
        for row in summaries.values()
        if row["ood_position_rmse"]["mean"] is not None
        and row["ood_position_rmse"]["mean"] > 0
    ]
    floor = min(y_positive) / 5 if y_positive else 1e-6
    palette = (
        "#94a3b8",
        "#64748b",
        "#a78bfa",
        "#7c3aed",
        "#2563eb",
        "#f97316",
        "#16a34a",
        "#be185d",
        "#78716c",
        "#111827",
        "#dc2626",
    )
    zero_shown = False
    for index, arm in enumerate(arms):
        row = summaries[arm]
        x, y = row["optimization_seconds"]["mean"], row["ood_position_rmse"]["mean"]
        if x is None or y is None:
            continue
        zero_shown |= y == 0
        axis.scatter(
            [x],
            [floor if y == 0 else y],
            color=palette[index % len(palette)],
            marker="v" if y == 0 else "o",
            s=65,
            label=arm + (" (exact zero)" if y == 0 else ""),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Mean cumulative optimization time (seconds)")
    axis.set_ylabel("Mean OOD position RMSE (positive values; log axis)")
    axis.set_title(f"Observed accuracy and training time\n{phase_label}", loc="left")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    note = "Means over paired units; time excludes validation/evaluation. This is not a general efficiency claim."
    if zero_shown:
        note += f"\nExact zeros are marked at {floor:.2g} solely for plotting; tables retain zero."
    figure.text(0.05, 0.02, note, fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    save(figure, "accuracy_time")


def render_analysis(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write deterministic numerical summaries and three PNG/PDF figures.

    Only the nine named analysis products are written, so this can regenerate
    analysis inside a retained phase directory without changing raw evidence.
    """
    analysis = analyze_phase(payload)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _figures(analysis, output)
    _tables(analysis, output / "tables.csv")
    (output / "results.md").write_text(_markdown(analysis), encoding="utf-8")
    analysis["artifacts"] = [
        "analysis.json",
        "results.md",
        "tables.csv",
        *[
            f"{stem}.{extension}"
            for stem in ("regime_heatmap", "paired_effects", "accuracy_time")
            for extension in ("png", "pdf")
        ],
    ]
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return analysis
