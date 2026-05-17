from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_benchmark_cli(tmp_path: Path):
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark.json"
    proc = subprocess.run([sys.executable, "scripts/benchmark.py", "--config", "configs/smoke.yaml", "--seeds", "3", "--modes", "base", "no_graph", "--steps", "3", "--output", str(output)], check=True, capture_output=True, text=True, cwd=repo, env=env)
    summary = json.loads(output.read_text())
    assert "runs" in summary
    assert "by_mode" in summary
    assert output.with_suffix(".csv").exists()
