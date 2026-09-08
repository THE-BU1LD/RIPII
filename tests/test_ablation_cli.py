from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_ablation_cli(tmp_path: Path):
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "ablation.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/ablate.py",
            "--config",
            "configs/mechanism_smoke.yaml",
            "--mode",
            "no_vq",
            "--output",
            str(output),
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    data = json.loads(output.read_text())
    assert data["mode"] == "no_vq"
    assert "summary" in data
