"""Hand-verifiable artificial measurements, never scientific training runs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ripii.world.closure_analysis import (
    OOD_SPLITS,
    SPLITS,
    _sign_flip_p,
    analyze_phase,
    render_analysis,
)


def _payload(phase: str = "confirmation", ratio: float = 0.9) -> dict:
    arm_ids = [
        "local_none_s1",
        "local_env_s1",
        "local_none_s4",
        "local_env_s4",
        "hier_env_s4",
        "global_env_s4",
        "mean_env_s4",
        "law_env_s4",
        "prior_env_s4",
        "oracle",
    ]
    seeds = [{"data": 100 + i, "model": 200 + i} for i in range(6)]
    protocol = {
        "format": "ripii-closure-study-protocol-v1",
        "study_id": "closure_study_v1",
        "dataset": {"evaluation_splits": list(SPLITS)},
        "arms": [{"id": arm} for arm in arm_ids],
        "compute_control": {"id": "global_compute"},
        "seeds": {"dry_run": seeds[0], "development": seeds[:3], "confirmation": seeds},
        "statistics": {
            "primary_candidate": "hier_env_s4",
            "primary_control": "global_compute",
            "practical_ratio": 0.95,
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": 46001,
            "interval_percentiles": [2.5, 97.5],
            "alpha": 0.05,
            "iid_max_ratio": 1.05,
            "required_other_controls": ["mean_env_s4", "law_env_s4"],
        },
    }
    selected = (
        [seeds[0]]
        if phase == "dry_run"
        else seeds[:3]
        if phase == "development"
        else seeds
    )
    units = []
    for index, seed in enumerate(selected):
        arms = {}
        for arm in [*arm_ids, "global_compute"]:
            base = float(index + 1)
            value = (
                0.0
                if arm == "oracle"
                else base * ratio
                if arm == "hier_env_s4"
                else base
            )
            parameters = (
                0
                if arm in ("oracle", "prior_env_s4")
                else 1
                if arm == "law_env_s4"
                else 100
            )
            seconds = (
                0.0 if parameters == 0 else 2.0 if arm == "global_compute" else 1.0
            )
            metrics = {
                split: {
                    "position_rmse": value,
                    "velocity_rmse": value * 2,
                    "position_rmse_h1": value / 2,
                    "position_rmse_h32": value,
                }
                for split in SPLITS
            }
            arms[arm] = {
                "parameters": parameters,
                "optimization_seconds": seconds,
                "completed_updates": 0 if parameters == 0 else 120,
                "compute_coverage_ratio": 2.0 if arm == "global_compute" else None,
                "metrics": metrics,
                "noise_metrics": {"position_rmse": value * 1.1},
                "profile": {f"seconds_per_forward_n{n}": n * 1e-4 for n in (2, 4, 8)},
            }
        units.append(
            {"data_seed": seed["data"], "model_seed": seed["model"], "arms": arms}
        )
    return {
        "format": "ripii-closure-phase-results-v1",
        "phase": phase,
        "protocol": protocol,
        "units": units,
        "complete": True,
    }


def test_constant_paired_benefit_matches_exact_sign_test_and_bootstrap():
    payload = _payload()
    unchanged = copy.deepcopy(payload)
    analysis = analyze_phase(payload)
    primary = analysis["comparisons"]["global_compute"]
    assert primary["geometric_ratio"] == pytest.approx(0.9)
    assert primary["interval_95"] == pytest.approx([0.9, 0.9])
    assert primary["primary_sign_flip_p"] == 2 / 64
    assert analysis["decision_h2"] == "SUPPORTED"
    assert (
        analysis["arm_summaries"]["global_compute"]["regimes"]["fast"]["position_rmse"][
            "mean"
        ]
        == 3.5
    )
    assert all(
        result["primary_sign_flip_p"] is None
        for name, result in analysis["comparisons"].items()
        if name != "global_compute"
    )
    assert payload == unchanged


def test_sign_flip_is_centered_on_practical_threshold_and_is_two_sided():
    assert _sign_flip_p([math.log(0.95)] * 6, math.log(0.95)) == 1.0
    assert _sign_flip_p([math.log(1.0)] * 6, math.log(0.95)) == 2 / 64
    assert (
        _sign_flip_p([math.log(0.95) - 0.2, math.log(0.95) + 0.2], math.log(0.95))
        == 1.0
    )


def test_ood_averaging_precedes_paired_ratio_and_bootstrap_is_reproducible():
    payload = _payload()
    for unit in payload["units"]:
        for split, left, right in zip(
            OOD_SPLITS, [1.0, 4.0, 1.0], [1.0, 1.0, 10.0], strict=True
        ):
            unit["arms"]["hier_env_s4"]["metrics"][split]["position_rmse"] = left
            unit["arms"]["global_compute"]["metrics"][split]["position_rmse"] = right
    first = analyze_phase(payload)
    second = analyze_phase(payload)
    assert first == second
    assert first["comparisons"]["global_compute"]["geometric_ratio"] == pytest.approx(
        0.5
    )
    assert first["comparisons"]["global_compute"]["n_pairs"] == 6
    assert first["unit_outcomes"][0]["arms"]["hier_env_s4"]["ood_position_rmse"] == 2.0


@pytest.mark.parametrize("phase", ["development", "dry_run"])
def test_nonconfirmation_never_gets_significance_or_supported_decision(phase):
    analysis = analyze_phase(_payload(phase))
    assert analysis["decision_h2"] == "NOT_TESTED"
    assert analysis["comparisons"]["global_compute"]["primary_sign_flip_p"] is None


@pytest.mark.parametrize(
    "change", ["incomplete", "missing_unit", "missing_arm", "missing_split"]
)
def test_incomplete_confirmation_is_inconclusive(change):
    payload = _payload(ratio=1.2)
    if change == "incomplete":
        payload["complete"] = False
    elif change == "missing_unit":
        payload["units"].pop()
    elif change == "missing_arm":
        del payload["units"][0]["arms"]["law_env_s4"]
    else:
        del payload["units"][0]["arms"]["hier_env_s4"]["metrics"]["fast"]
    analysis = analyze_phase(payload)
    assert analysis["decision_h2"] == "INCONCLUSIVE"
    assert analysis["validation"]["missing"]
    assert analysis["comparisons"]["global_compute"]["primary_sign_flip_p"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_rmse_is_retained_and_cannot_become_negative(value):
    payload = _payload(ratio=1.2)
    payload["units"][0]["arms"]["hier_env_s4"]["metrics"]["fast"]["position_rmse"] = (
        value
    )
    analysis = analyze_phase(payload)
    assert analysis["decision_h2"] == "INVALID"
    assert analysis["validation"]["errors"]
    json.dumps(analysis, allow_nan=False)


def test_zero_learned_error_does_not_create_a_pseudocount_or_selected_subset():
    payload = _payload()
    for split in OOD_SPLITS:
        payload["units"][0]["arms"]["law_env_s4"]["metrics"][split]["position_rmse"] = (
            0.0
        )
    analysis = analyze_phase(payload)
    assert analysis["decision_h2"] == "INCONCLUSIVE"
    assert analysis["comparisons"]["law_env_s4"]["geometric_ratio"] is None
    assert analysis["comparisons"]["law_env_s4"]["interval_95"] is None
    assert analysis["arm_summaries"]["oracle"]["ood_position_rmse"]["mean"] == 0.0


def test_compute_cap_failure_overrides_otherwise_negative_comparison():
    payload = _payload(ratio=1.2)
    row = payload["units"][0]["arms"]["global_compute"]
    row.update(optimization_seconds=0.9, compute_coverage_ratio=0.9)
    analysis = analyze_phase(payload)
    assert analysis["decision_h2"] == "INCONCLUSIVE"
    assert (
        analysis["comparisons"]["law_env_s4"]["practical_evidence"]
        == "INTERVAL_AT_OR_ABOVE_THRESHOLD"
    )
    assert not analysis["compute_coverage"]["complete"]


@pytest.mark.parametrize(
    "change", ["coverage_mismatch", "parameters", "duplicate", "replacement"]
)
def test_inconsistent_design_or_unit_identity_is_invalid(change):
    payload = _payload()
    if change == "coverage_mismatch":
        payload["units"][0]["arms"]["global_compute"]["compute_coverage_ratio"] = 4.0
    elif change == "parameters":
        payload["units"][0]["arms"]["global_compute"]["parameters"] = 101
    elif change == "duplicate":
        payload["units"].append(copy.deepcopy(payload["units"][0]))
    else:
        payload["units"][0]["data_seed"] = 9999
    analysis = analyze_phase(payload)
    assert analysis["decision_h2"] == "INVALID"


@pytest.mark.parametrize("ratio", [0.95, 1.0, 1.2])
def test_required_control_can_trigger_frozen_negative_rule(ratio):
    analysis = analyze_phase(_payload(ratio=ratio))
    assert analysis["decision_h2"] == "NEGATIVE"
    assert "global_compute" in analysis["h2"]["controls_at_or_above_practical_bound"]


def test_iid_regression_blocks_support_without_inventing_negative_rule():
    payload = _payload()
    for unit in payload["units"]:
        unit["arms"]["hier_env_s4"]["metrics"]["test"]["position_rmse"] = (
            2 * unit["arms"]["global_compute"]["metrics"]["test"]["position_rmse"]
        )
    analysis = analyze_phase(payload)
    assert analysis["decision_h2"] == "INCONCLUSIVE"
    assert not analysis["h2"]["safeguards"]["iid_point_at_most_105"]
    assert analysis["h2"]["descriptive_mixed_regime_direction"]


def test_fixed_bootstrap_resamples_only_independent_units():
    payload = _payload()
    ratios = np.asarray([0.5, 0.7, 0.8, 0.9, 1.1, 1.3])
    for unit, ratio in zip(payload["units"], ratios, strict=True):
        for split in OOD_SPLITS:
            unit["arms"]["hier_env_s4"]["metrics"][split]["position_rmse"] = float(
                ratio
            )
            unit["arms"]["global_compute"]["metrics"][split]["position_rmse"] = 1.0
    rng = np.random.default_rng(46001)
    samples = rng.integers(0, 6, size=(10000, 6))
    expected = np.percentile(
        np.exp(np.mean(np.log(ratios)[samples], axis=1)), [2.5, 97.5]
    )
    analysis = analyze_phase(payload)
    assert analysis["comparisons"]["global_compute"]["interval_95"] == pytest.approx(
        expected
    )


def test_protocol_constants_and_six_confirmation_units_are_enforced():
    payload = _payload()
    payload["protocol"]["statistics"]["bootstrap_repetitions"] = 999
    with pytest.raises(ValueError, match="statistical constants"):
        analyze_phase(payload)
    payload = _payload()
    payload["protocol"]["seeds"]["confirmation"].pop()
    with pytest.raises(ValueError, match="exactly 6"):
        analyze_phase(payload)


def test_render_writes_numerical_evidence_and_real_figures(tmp_path: Path):
    payload = _payload("development")
    unrelated = tmp_path / "raw_predictions.bin"
    unrelated.write_bytes(b"retained raw evidence")
    analysis = render_analysis(payload, tmp_path)
    assert unrelated.read_bytes() == b"retained raw evidence"
    assert len(analysis["artifacts"]) == 9
    retained = json.loads((tmp_path / "analysis.json").read_text())
    assert retained == analysis
    assert "NOT_TESTED" in (tmp_path / "results.md").read_text()
    assert "exact zero" in (tmp_path / "results.md").read_text().lower()
    assert "arithmetic_mean" in (tmp_path / "tables.csv").read_text()
    for stem in ("regime_heatmap", "paired_effects", "accuracy_time"):
        assert (tmp_path / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
