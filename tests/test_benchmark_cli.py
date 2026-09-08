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
    retained = tmp_path / "retained"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--config",
            "configs/mechanism_smoke.yaml",
            "--seeds",
            "3",
            "--modes",
            "base",
            "no_graph",
            "no_vq",
            "--steps",
            "3",
            "--output",
            str(output),
            "--retain-run-dir",
            str(retained),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
    )
    summary = json.loads(output.read_text())
    assert "runs" in summary
    assert "by_mode" in summary
    assert output.with_suffix(".csv").exists()
    assert output.with_suffix(".manifest.json").exists()
    assert summary["evidence_status"] == "development_only"
    assert summary["profile"] == "mechanism_smoke"
    assert summary["paired_design"]["shared_initial_state"] is True
    assert len({row["initialization"]["sha256"] for row in summary["runs"]}) == 1
    assert {row["mode"] for row in summary["runs"]} == {
        "base",
        "no_graph",
        "no_vq",
    }
    no_vq = next(row for row in summary["runs"] if row["mode"] == "no_vq")
    assert no_vq["quantizer_metrics_applicable"] is False
    assert no_vq["perplexity_coarse"] is None
    assert "perplexity_coarse_mean" not in summary["by_mode"]["no_vq"]
    assert "quantizer_metrics_applicable_mean" not in summary["by_mode"]["no_vq"]
    assert (retained / "_initial_states" / "seed_3.pt").is_file()
    assert (retained / "base" / "seed_3" / "final.pt").is_file()
    assert not (retained / "base" / "seed_3" / "latest.pt").exists()
    assert not (retained / "base" / "seed_3" / "best.pt").exists()
