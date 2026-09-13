from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path("scripts/index_runs.py")
    spec = importlib.util.spec_from_file_location("index_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_inventory_preserves_failures_and_manifest_status(tmp_path):
    module = _module()
    failed = tmp_path / "failed_study"
    failed.mkdir()
    (failed / "launch.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    complete = tmp_path / "complete_study"
    complete.mkdir()
    (complete / "manifest.json").write_text("{}\n", encoding="utf-8")
    quarantined = tmp_path / "base.incomplete.20260913"
    quarantined.mkdir()
    result = module.inventory(tmp_path)
    statuses = {row["run"]: row["status"] for row in result["runs"]}
    assert statuses == {
        "base.incomplete.20260913": "quarantined_incomplete",
        "complete_study": "manifest_present",
        "failed_study": "failed",
    }
    assert result["claim_boundary"].startswith("inventory only")
