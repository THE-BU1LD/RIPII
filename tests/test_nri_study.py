from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_nri_external_study import (
    FROZEN_SEEDS,
    _signature,
    frozen_experiment,
    run,
    verify_capsule,
)


def test_nri_study_rejects_protocol_drift_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="frozen NRI"):
        run(
            tmp_path / "data",
            output,
            replace(frozen_experiment(), steps=99),
            FROZEN_SEEDS,
        )
    assert not output.exists()


def test_nri_capsule_signature_and_retained_hashes_fail_closed(tmp_path: Path) -> None:
    source = Path("research/results/development/nri_external_development_v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / "capsule.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_capsule(path)["status"] == "PASS"
    payload["retained"]["report.md"]["content_text"] = "tampered\n"
    payload["result_sha256"] = _signature(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="retained content mismatch"):
        verify_capsule(path)


def test_retained_nri_capsule_has_semantic_integrity() -> None:
    path = Path("research/results/development/nri_external_development_v1.json")
    assert verify_capsule(path)["status"] == "PASS"


def test_nri_capsule_rejects_a_resigned_false_decision(tmp_path: Path) -> None:
    source = Path("research/results/development/nri_external_development_v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    summary = json.loads(payload["retained"]["summary.json"]["content_text"])
    summary["decision"] = "development_gate_pass"
    content = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    payload["retained"]["summary.json"] = {
        "content_text": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "bytes": len(content.encode()),
    }
    payload["result_sha256"] = _signature(payload)
    path = tmp_path / "resigned-false-decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision is inconsistent"):
        verify_capsule(path)
