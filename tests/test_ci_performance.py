from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_ci_performance import check

ROOT = Path(__file__).resolve().parents[1]


def test_ci_performance_contract_passes() -> None:
    result = check(ROOT / "research/protocols/ci_performance_contract_v1.json")
    assert result["status"] == "PASS"
    assert result["parameters"] <= 5000


def test_ci_performance_contract_fails_closed(tmp_path: Path) -> None:
    source = ROOT / "research/protocols/ci_performance_contract_v1.json"
    contract = json.loads(source.read_text(encoding="utf-8"))
    contract["max_trainable_parameters"] = 1
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parameter budget exceeded"):
        check(path)
