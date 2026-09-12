from __future__ import annotations

import json

import pytest

from scripts.profile_world_efficiency import _signature, verify_output


def _payload() -> dict:
    payload = {
        "format": "ripii-world-suite-profile-v1",
        "rows": [
            {
                "profile": {
                    "timing_seconds": {"mean": 0.01, "median": 0.009},
                    "scene_steps_per_second_at_mean": 800.0,
                }
            }
        ],
    }
    payload["signature"] = _signature(payload)
    return payload


def test_suite_profile_verification_fails_closed(tmp_path) -> None:
    artifact = tmp_path / "profile.json"
    payload = _payload()
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_output(artifact)["rows_verified"] == 1

    payload["rows"][0]["profile"]["timing_seconds"]["mean"] = float("inf")
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_output(artifact)


def test_suite_profile_rejects_tampering(tmp_path) -> None:
    artifact = tmp_path / "profile.json"
    payload = _payload()
    payload["rows"][0]["profile"]["timing_seconds"]["mean"] = 0.02
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        verify_output(artifact)
