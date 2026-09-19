from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.run_ripii_mr_matrix import (
    PROVENANCE_PATHS,
    SOURCE_ROOT,
    VARIANTS,
    _manifest,
    analyze,
    sha256_file,
    verify_matrix,
)


def _write_child(output: Path, data_seed: int, improvement: float) -> None:
    rows = []
    for seed in (131, 137, 139, 149, 151):
        for control in VARIANTS[1:]:
            rows.append(
                {
                    "candidate": "ripii_mr",
                    "control": control,
                    "seed": seed,
                    "ood_relative_improvement": improvement,
                    "id_relative_regression": 0.0,
                    "passes": improvement >= 0.05,
                }
            )
    child = output / f"data_seed_{data_seed}"
    child.mkdir(parents=True)
    (child / "summary.json").write_text(
        json.dumps(
            {
                "evidence_status": "development_only",
                "paired_comparisons": rows,
            }
        ),
        encoding="utf-8",
    )


def test_matrix_analysis_uses_all_data_model_and_control_pairs(tmp_path: Path) -> None:
    for data_seed in (3101, 3203, 3307):
        _write_child(tmp_path, data_seed, 0.06)
    result = analyze(tmp_path, (3101, 3203, 3307))
    assert result["decision"] == "advance_development"
    assert len(result["paired_comparisons"]) == 60
    assert all(row["all_cells_pass"] for row in result["by_control"].values())


def test_matrix_analysis_preserves_a_single_negative_cell(tmp_path: Path) -> None:
    for data_seed in (3101, 3203):
        _write_child(tmp_path, data_seed, 0.06)
    _write_child(tmp_path, 3307, 0.04)
    result = analyze(tmp_path, (3101, 3203, 3307))
    assert result["decision"] == "no_advance"
    assert not all(row["all_cells_pass"] for row in result["by_control"].values())


def test_matrix_analysis_recomputes_the_gate_instead_of_trusting_child(
    tmp_path: Path,
) -> None:
    for data_seed in (3101, 3203, 3307):
        _write_child(tmp_path, data_seed, 0.04)
        child = tmp_path / f"data_seed_{data_seed}" / "summary.json"
        payload = json.loads(child.read_text(encoding="utf-8"))
        for row in payload["paired_comparisons"]:
            row["passes"] = True
        child.write_text(json.dumps(payload), encoding="utf-8")
    result = analyze(tmp_path, (3101, 3203, 3307))
    assert result["decision"] == "no_advance"
    assert not any(row["passes"] for row in result["paired_comparisons"])


def test_matrix_manifest_detects_changed_artifact(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"decision": "no_advance"}), encoding="utf-8"
    )
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    _manifest(tmp_path)
    assert verify_matrix(tmp_path)["status"] == "PASS"
    (tmp_path / "protocol.json").write_text('{"changed": true}', encoding="utf-8")
    try:
        verify_matrix(tmp_path)
    except ValueError as exc:
        assert "changed artifact" in str(exc)
    else:
        raise AssertionError("changed artifact was accepted")


def test_matrix_manifest_detects_unexpected_artifact(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"decision": "no_advance"}), encoding="utf-8"
    )
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    _manifest(tmp_path)
    (tmp_path / "untracked.txt").write_text("not in manifest", encoding="utf-8")
    try:
        verify_matrix(tmp_path)
    except ValueError as exc:
        assert "unmanifested" in str(exc)
    else:
        raise AssertionError("unexpected artifact was accepted")


def test_matrix_manifest_binds_runner_analysis_and_protocol_sources(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"decision": "no_advance"}), encoding="utf-8"
    )
    (output / "protocol.json").write_text("{}", encoding="utf-8")
    source_root = tmp_path / "source"
    for relative in PROVENANCE_PATHS:
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / relative, destination)

    _manifest(output, source_root=source_root)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "ripii-mr-development-manifest-v2"
    assert manifest["protocol_sha256"] == sha256_file(output / "protocol.json")
    assert set(manifest["source_sha256"]) == set(PROVENANCE_PATHS)
    assert verify_matrix(output, source_root=source_root)["status"] == "PASS"

    for relative in PROVENANCE_PATHS:
        source = source_root / relative
        original = source.read_bytes()
        source.write_bytes(original + b"\n# provenance tamper\n")
        try:
            verify_matrix(output, source_root=source_root)
        except ValueError as exc:
            assert "changed matrix source" in str(exc)
            assert relative in str(exc)
        else:
            raise AssertionError(f"changed provenance source was accepted: {relative}")
        source.write_bytes(original)


def test_matrix_manifest_accepts_legacy_v1_without_source_binding(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"decision": "no_advance"}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ripii-mr-development-manifest-v1",
                "artifacts": [
                    {
                        "path": "summary.json",
                        "bytes": summary.stat().st_size,
                        "sha256": sha256_file(summary),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = verify_matrix(tmp_path)
    assert result == {"status": "PASS", "decision": "no_advance", "artifacts": 1}
