from __future__ import annotations

import json

import pytest

from scripts.analyze_objective_study import _signature, verify_output


def test_objective_analysis_verifier_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "analysis.json"
    payload = {
        "format": "ripii-objective-study-analysis-v1",
        "comparisons": {
            "base": {
                "relative_reduction_by_seed": [-0.1, -0.2],
                "all_seed_five_percent_rule": False,
            }
        },
        "supported_additions": [],
        "decision": "no_auxiliary_advance",
    }
    payload["signature"] = _signature(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_output(path)["comparisons_verified"] == 1
    payload["comparisons"]["base"]["all_seed_five_percent_rule"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        verify_output(path)
