from __future__ import annotations

from pathlib import Path


def test_source_manifest_excludes_local_run_trees() -> None:
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")
    for relative in (
        "research/results/pilot_v1/runs",
        "research/results/pilot_v2/runs",
        "research/results/development/objective_study_v1/runs",
    ):
        assert f"prune {relative}" in manifest
