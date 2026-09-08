from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_full_cli_pipeline(tmp_path: Path):
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    config = Path("configs/smoke.yaml").resolve()
    out_dir = tmp_path / "runs"
    env["RIPII_OUTPUT_DIR"] = str(out_dir)
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config)],
        check=True,
        env=env,
        cwd=repo,
    )
    checkpoint = out_dir / "final.pt"
    assert checkpoint.exists()
    eval_proc = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )
    summary = json.loads(eval_proc.stdout)
    assert "total" in summary
    assert summary["evidence_status"] == "development_only"
    assert summary["quantizer_metrics_applicable"] is False
    assert summary["perplexity_coarse"] is None
    assert summary["usage"] is None
    assert (out_dir / "eval.json").exists()
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnostics.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out_dir / "diagnostics.png"),
        ],
        check=True,
        cwd=repo,
    )
    assert (out_dir / "diagnostics_depth.png").exists()
    assert (out_dir / "diagnostics_usage.png").exists()
    assert (out_dir / "diagnostics_geometry.png").exists()
    assert not (repo / "runs" / "ripii_smoke" / "diagnostics_gate.png").exists()
