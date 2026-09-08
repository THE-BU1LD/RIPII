from __future__ import annotations

import json

import pytest

from ripii.utils.power import approximate_paired_seed_count
from scripts.plan_power import _signature, verify_plan_artifact


def test_power_plan_uses_development_variance_and_effect_threshold() -> None:
    result = approximate_paired_seed_count(
        [-0.1, -0.05, 0.0, 0.05, 0.1],
        minimum_detectable_effect=0.05,
    )
    assert result["recommended_minimum_pairs"] >= 6
    assert result["minimum_detectable_effect"] == 0.05
    assert result["development_pairs"] == 5
    with pytest.raises(ValueError, match="zero"):
        approximate_paired_seed_count([0.1, 0.1], minimum_detectable_effect=0.05)
    with pytest.raises(ValueError, match="invalid"):
        approximate_paired_seed_count([0.1], minimum_detectable_effect=0.05)


def test_power_plan_artifact_signature_fails_closed(tmp_path) -> None:
    path = tmp_path / "plan.json"
    payload = {
        "format": "ripii-paired-power-plan-v1",
        "recommended_minimum_pairs": 10,
        "minimum_detectable_effect": 0.05,
        "evidence_status": "prospective_planning_from_development_variance",
    }
    payload["signature"] = _signature(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_plan_artifact(path)["status"] == "PASS"
    payload["recommended_minimum_pairs"] = 9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        verify_plan_artifact(path)
