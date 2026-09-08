from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_artifact_detects_tampering(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    artifact = tmp_path / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": artifact.name,
                        "sha256": _sha256(artifact),
                        "size": artifact.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "scripts/verify_artifact.py",
        "--manifest",
        str(manifest),
    ]
    valid = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    assert valid.returncode == 0
    assert json.loads(valid.stdout)["status"] == "PASS"

    artifact.write_text("tampered", encoding="utf-8")
    invalid = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    assert invalid.returncode == 1
    assert "SHA-256 mismatch" in invalid.stdout


def test_portable_summary_is_explicit_and_still_detects_tampering(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    artifacts = []
    for name, content in (
        ("summary.json", "{}\n"),
        ("summary.csv", "metric,value\nloss,1\n"),
        ("summary.md", "# Summary\n"),
    ):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            {
                "relative_path": name,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )
    artifacts.append(
        {
            "relative_path": "runs/base/seed_1/final.pt",
            "sha256": "0" * 64,
            "size": 1,
        }
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    command = [
        sys.executable,
        "scripts/verify_artifact.py",
        "--manifest",
        str(manifest),
        "--portable-summary",
    ]

    valid = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    result = json.loads(valid.stdout)
    assert valid.returncode == 0
    assert result["status"] == "PASS"
    assert result["verification_scope"] == "portable_summary_only"
    assert result["artifacts_verified"] == 3
    assert result["run_artifacts_explicitly_skipped"] == 1

    (tmp_path / "summary.md").write_text("tampered\n", encoding="utf-8")
    invalid = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    assert invalid.returncode == 1
    assert "SHA-256 mismatch: summary.md" in invalid.stdout
