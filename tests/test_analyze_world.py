from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_world import _load, analyze

SPLITS = ("test", "more_objects", "composition", "fast")


def _row(variant: str, seed: int, value: float = 1.0) -> dict:
    return {
        "variant": variant,
        "bottleneck": "continuous",
        "seed": seed,
        "metrics": {
            split: {"position_rmse": value + offset / 10}
            for offset, split in enumerate(SPLITS)
        },
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"runs": rows}), encoding="utf-8")
    return path


def test_analysis_records_portable_provenance(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "summary.json",
        [_row("multiscale", 1, 0.9), _row("graph", 1, 1.0)],
    )
    result = analyze(
        source,
        candidate="multiscale",
        baseline="graph",
        bottleneck="continuous",
    )
    assert result["input"]["name"] == "summary.json"
    assert "/" not in result["input"]["name"]
    assert len(result["input"]["sha256"]) == 64
    assert set(result["source_sha256"]) == {
        "scripts/analyze_world.py",
        "ripii/utils/statistics.py",
    }


def test_analysis_rejects_duplicate_and_unpaired_seeds(tmp_path: Path) -> None:
    duplicate = _write(
        tmp_path / "duplicate.json",
        [_row("multiscale", 1), _row("multiscale", 1), _row("graph", 1)],
    )
    with pytest.raises(ValueError, match="duplicate"):
        analyze(
            duplicate,
            candidate="multiscale",
            baseline="graph",
            bottleneck="continuous",
        )
    unpaired = _write(
        tmp_path / "unpaired.json",
        [_row("multiscale", 1), _row("graph", 2)],
    )
    with pytest.raises(ValueError, match="seed sets differ"):
        analyze(
            unpaired,
            candidate="multiscale",
            baseline="graph",
            bottleneck="continuous",
        )


def test_analysis_verifies_capsule_before_using_retained_summary(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = repository / "research/results/development/world_v3_convergence_capsule_v2.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["retained"]["summary.json"]["content_text"] += " "
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        _load(tampered)
