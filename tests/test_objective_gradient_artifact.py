from __future__ import annotations

import json

import pytest

from scripts.audit_objective_gradients import _signature, verify_output


def test_objective_gradient_artifact_fails_closed(tmp_path) -> None:
    path = tmp_path / "audit.json"
    payload = {
        "format": "ripii-objective-gradient-audit-v1",
        "rows": [
            {
                "diagnostic": {
                    "terms": {"recon": {"weighted_gradient_l2": 1.0}}
                }
            }
        ],
    }
    payload["signature"] = _signature(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_output(path)["seeds_verified"] == 1
    payload["rows"][0]["diagnostic"]["terms"]["recon"][
        "weighted_gradient_l2"
    ] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        verify_output(path)
