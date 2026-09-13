from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path("scripts/run_post_correction.py")
    spec = importlib.util.spec_from_file_location("run_post_correction", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_post_correction_preflight_reports_frozen_cell_count():
    proc = subprocess.run(
        [sys.executable, "scripts/run_post_correction.py", "--preflight"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "READY"' in proc.stdout
    assert '"cells": 180' in proc.stdout


def test_post_correction_manifest_verification(tmp_path):
    module = _module()
    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": artifact.name,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "size": artifact.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    module.verify_manifest(manifest)
    artifact.write_text("tampered\n", encoding="utf-8")
    try:
        module.verify_manifest(manifest)
    except ValueError as exc:
        assert "integrity failure" in str(exc)
    else:
        raise AssertionError("tampered artifact was accepted")


def test_post_correction_analysis_is_complete_and_predeclared():
    module = _module()
    rows = []
    for mode in module.MODES:
        for seed in module.SEEDS:
            recon = 0.8 if mode == "base" else 1.0
            rows.append({"mode": mode, "seed": seed, "recon": recon})
    result = module.analyze(rows)
    assert result["decision"] == "development_gate_pass"
    assert len(result["comparisons_vs_base"]) == len(module.MODES) - 1
    rows.pop()
    with pytest.raises(ValueError, match="grid"):
        module.analyze(rows)


def test_calibration_uses_repeated_monotone_upper_bound():
    module = _module()
    measurements = [
        {"steps": 60, "seconds": 12.0},
        {"steps": 60, "seconds": 14.0},
        {"steps": 180, "seconds": 26.0},
        {"steps": 180, "seconds": 28.0},
        {"steps": 300, "seconds": 40.0},
        {"steps": 300, "seconds": 44.0},
    ]
    result = module.estimate_calibration(measurements, target_steps=600)
    assert result["seconds_per_step_upper_bound"] > 0
    assert result["estimated_seconds_per_target_cell"] >= 84.0
    assert len(result["median_measurements"]) == 3


def test_calibration_rejects_timing_inversion_from_failed_overnight_pattern():
    module = _module()
    measurements = [
        {"steps": 60, "seconds": 59.0},
        {"steps": 60, "seconds": 58.0},
        {"steps": 180, "seconds": 50.0},
        {"steps": 180, "seconds": 49.0},
        {"steps": 300, "seconds": 45.0},
        {"steps": 300, "seconds": 44.0},
    ]
    with pytest.raises(RuntimeError, match="do not increase"):
        module.estimate_calibration(measurements)


def test_calibration_rejects_more_workers_than_frozen_seeds(tmp_path):
    module = _module()
    with pytest.raises(ValueError, match="seed-count workers"):
        module.calibrate(tmp_path, tmp_path / "config.yaml", workers=6)


def test_exact_directory_manifest_detects_extra_file(tmp_path):
    module = _module()
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    module.manifest_directory(tmp_path, manifest)
    module.verify_manifest(manifest)
    (tmp_path / "unexpected.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact set mismatch"):
        module.verify_manifest(manifest)


def test_disk_guard_is_fail_closed(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "free_gib", lambda _path: 1.5)
    with pytest.raises(RuntimeError, match="at least 2.00 GiB"):
        module.require_disk_space(tmp_path, 2.0)


def test_cell_runner_retries_logs_and_then_resumes(tmp_path, monkeypatch):
    module = _module()
    output = tmp_path / "study"
    output.mkdir()
    launch_path = output / "launch.json"
    launch = {"status": "running", "events": []}
    module.write_json(launch_path, launch)
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=17)
        benchmark = Path(command[command.index("--output") + 1])
        benchmark.write_text(
            json.dumps({"runs": [{"mode": "base", "seed": 1009}]}),
            encoding="utf-8",
        )
        artifact = {
            "relative_path": "benchmark.json",
            "sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            "size": benchmark.stat().st_size,
        }
        (benchmark.parent / "manifest.json").write_text(
            json.dumps({"artifacts": [artifact]}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "require_disk_space", lambda *_args: 100.0)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen\n", encoding="utf-8")
    module.run_cell(
        output=output,
        config=tmp_path / "config.yaml",
        protocol=protocol,
        protocol_sha=module.sha256(protocol),
        mode="base",
        seed=1009,
        retries=1,
        minimum_free_gib=1,
        launch=launch,
        launch_path=launch_path,
    )
    assert calls == 2
    assert any(row["event"] == "cell_attempt_failed" for row in launch["events"])
    assert any(row["event"] == "cell_complete" for row in launch["events"])
    assert len(list((output / "_logs").glob("*.log"))) == 2
    module.run_cell(
        output=output,
        config=tmp_path / "config.yaml",
        protocol=protocol,
        protocol_sha=module.sha256(protocol),
        mode="base",
        seed=1009,
        retries=1,
        minimum_free_gib=1,
        launch=launch,
        launch_path=launch_path,
    )
    assert calls == 2
