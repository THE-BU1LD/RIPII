from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_report_cli(tmp_path: Path):
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    repo = Path(__file__).resolve().parents[1]
    benchmark = tmp_path / "benchmark.json"
    payload = {
        "runs": [{"mode": "base", "seed": 3, "total": 1.0, "recon": 0.5}],
        "by_mode": {"base": {"total_mean": 1.0, "recon_mean": 0.5}},
    }
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "report.md"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/report.py",
            "--benchmark",
            str(benchmark),
            "--output",
            str(output),
            "--summary-json",
            str(summary),
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    assert output.exists()
    assert summary.exists()
    data = json.loads(summary.read_text())
    assert "modes" in data
    report = output.read_text(encoding="utf-8")
    assert "cross-ablation ranking metric" in report
    assert "Ranking by total loss" not in report
