from __future__ import annotations

import json

import pytest
import torch

from ripii.world.experiment import Persistence
from ripii.world.failure_analysis import (
    evaluate_failure_regimes,
    transition_regime_masks,
)
from scripts.analyze_failures import (
    _analysis_signature,
    _source_is_ast_equivalent,
    verify_analysis,
)


def _data() -> dict[str, torch.Tensor]:
    states = torch.zeros(1, 2, 5, 6)
    mask = torch.tensor([[True, True, False, False, False]])
    states[..., 4] = 0.1
    states[..., 5] = 1.0
    states[0, :, 0, :2] = torch.tensor([0.0, 0.0])
    states[0, :, 1, :2] = torch.tensor([0.19, 0.0])
    actions = torch.zeros(1, 1, 5, 2)
    actions[0, 0, 0, 0] = 1.0
    return {"states": states, "actions": actions, "mask": mask}


def test_regimes_detect_contact_force_and_masked_slots() -> None:
    data = _data()
    regimes = transition_regime_masks(
        data["states"][:, :-1], data["actions"], data["mask"]
    )
    assert int(regimes["contact"].sum()) == 2
    assert int(regimes["forced"].sum()) == 1
    assert int(regimes["all"].sum()) == 2
    assert not regimes["free_flight"].any()
    result = evaluate_failure_regimes(Persistence(), data)
    assert result["contact"]["object_transitions"] == 2
    assert result["contact"]["position_rmse"] == 0.0


def test_failure_analysis_fails_closed() -> None:
    data = _data()
    data["actions"][0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        evaluate_failure_regimes(Persistence(), data)
    with pytest.raises(ValueError):
        transition_regime_masks(
            torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, dtype=torch.bool)
        )


def test_source_equivalence_allows_formatting_but_rejects_code_drift(
    tmp_path,
) -> None:
    retained = tmp_path / "retained.py"
    current = tmp_path / "current.py"
    retained.write_text("value = call(1, 2)\n", encoding="utf-8")
    current.write_text("value = call(\n    1,\n    2,\n)\n", encoding="utf-8")
    assert _source_is_ast_equivalent(current, retained)
    current.write_text("value = call(1, 3)\n", encoding="utf-8")
    assert not _source_is_ast_equivalent(current, retained)


def test_failure_analysis_signature_and_schema_fail_closed(tmp_path) -> None:
    path = tmp_path / "analysis.json"
    payload = {
        "format": "ripii-world-failure-analysis-v1",
        "full_run_verification": {
            "status": "PASS",
            "artifacts_verified": 3,
            "unexpected_files": 0,
        },
        "post_analysis_run_verification": {
            "status": "PASS",
            "artifacts_verified": 3,
            "unexpected_files": 0,
        },
        "checkpoint_model_source": {
            "exact_match": True,
            "ast_equivalent": True,
        },
        "rows": [{"variant": "graph", "seed": 3}],
    }
    payload["signature"] = _analysis_signature(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_analysis(path)["rows_verified"] == 1
    payload["rows"][0]["seed"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_analysis(path)
