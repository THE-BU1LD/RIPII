from __future__ import annotations

import hashlib
import json

import pytest

from scripts.run_coupling_study import (
    analyze,
    capture,
    verify,
    verify_capsule,
    write_manifest,
)


def _summary(multiscale: list[float], global_pool: list[float]) -> dict:
    rows = []
    for variant, values in (
        ("multiscale", multiscale),
        ("global_pool", global_pool),
    ):
        rows.extend(
            {
                "variant": variant,
                "seed": seed,
                "metrics": {"more_objects": {"position_rmse": value}},
            }
            for seed, value in zip((3, 7, 11), values)
        )
    return {"runs": rows}


def test_coupling_analysis_requires_hierarchy_specific_effect() -> None:
    local = _summary([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    coupled = _summary([0.8, 0.8, 0.8], [1.0, 1.0, 1.0])
    result = analyze({"local": local, "coupled": coupled})
    assert result["decision"] == "advance_conditional_hierarchy"
    assert result["coupling_interaction_mean"] == pytest.approx(0.2)
    no_effect = analyze({"local": local, "coupled": local})
    assert no_effect["decision"] == "no_advance"


def test_coupling_analysis_fails_on_missing_or_unpaired_regimes() -> None:
    local = _summary([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="exactly local and coupled"):
        analyze({"local": local})
    local["runs"] = [
        row
        for row in local["runs"]
        if not (row["variant"] == "global_pool" and row["seed"] == 3)
    ]
    with pytest.raises(ValueError, match="same paired seeds"):
        analyze({"local": local, "coupled": _summary([1, 1, 1], [1, 1, 1])})

    different_seeds = _summary([1, 1, 1], [1, 1, 1])
    for row in different_seeds["runs"]:
        row["seed"] += 1
    with pytest.raises(ValueError, match="local and coupled"):
        analyze(
            {
                "local": _summary([1, 1, 1], [1, 1, 1]),
                "coupled": different_seeds,
            }
        )

    duplicate = _summary([1, 1, 1], [1, 1, 1])
    duplicate["runs"].append(dict(duplicate["runs"][0]))
    with pytest.raises(ValueError, match="duplicate result cell"):
        analyze({"local": duplicate, "coupled": _summary([1, 1, 1], [1, 1, 1])})

    zero_control = _summary([1, 1, 1], [0, 1, 1])
    with pytest.raises(FloatingPointError, match="controls positive"):
        analyze({"local": zero_control, "coupled": zero_control})


def test_coupling_study_manifest_and_capsule_fail_closed(tmp_path) -> None:
    directory = tmp_path / "study"
    for relative in (
        "protocol.json",
        "summary.json",
        "report.md",
        "local/summary.json",
        "coupled/summary.json",
        "status.json",
    ):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "protocol.json":
            path.write_text('{"test": true}\n', encoding="utf-8")
        elif relative == "status.json":
            from ripii.world.run_status import RunTracker

            tracker = RunTracker.create(
                directory,
                run_kind="test",
                protocol_sha256=hashlib.sha256(
                    (directory / "protocol.json").read_bytes()
                ).hexdigest(),
            )
            tracker.transition("running")
            tracker.transition("complete")
        elif relative.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text(f"content for {relative}\n", encoding="utf-8")
    protocol_sha256 = hashlib.sha256(
        (directory / "protocol.json").read_bytes()
    ).hexdigest()
    (directory / "summary.json").write_text(
        json.dumps({"protocol_sha256": protocol_sha256}) + "\n", encoding="utf-8"
    )
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path.relative_to(directory)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
    (directory / "manifest.json").write_text(
        json.dumps({"format": "ripii-coupling-study-v1", "artifacts": artifacts}),
        encoding="utf-8",
    )
    assert verify(directory)["status"] == "PASS"
    capsule = tmp_path / "capsule.json"
    capture(directory, capsule)
    assert verify_capsule(capsule)["content_hashes_verified"] is True
    payload = json.loads(capsule.read_text(encoding="utf-8"))
    payload["evidence_status"] = "confirmatory"
    capsule.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_capsule(capsule)
    (directory / "report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="size or symlink|hash mismatch"):
        verify(directory)


def test_coupling_verifiers_reject_hashed_nonfinite_json(tmp_path) -> None:
    directory = tmp_path / "study"
    directory.mkdir()
    invalid = directory / "summary.json"
    invalid.write_text('{"value": 1e999}\n', encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ripii-coupling-study-v1",
                "artifacts": [
                    {
                        "path": "summary.json",
                        "sha256": hashlib.sha256(invalid.read_bytes()).hexdigest(),
                        "bytes": invalid.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite JSON"):
        verify(directory)


def test_master_manifest_includes_nested_manifests(tmp_path) -> None:
    directory = tmp_path / "study"
    nested = directory / "local/manifest.json"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        json.dumps({"format": "ripii-world-manifest-v1", "artifacts": []}) + "\n",
        encoding="utf-8",
    )
    write_manifest(directory)
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert [row["path"] for row in payload["artifacts"]] == ["local/manifest.json"]
    assert verify(directory)["status"] == "PASS"
