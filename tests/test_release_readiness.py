from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_release_readiness import check


def test_release_readiness_reports_owner_blockers(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as caught:
        check(tmp_path)
    payload = json.loads(str(caught.value))
    assert payload["status"] == "BLOCKED"
    assert "LICENSE" in payload["missing"]
    assert "release/OWNER_METADATA.json" in payload["missing"]


def test_release_readiness_accepts_complete_fixture(tmp_path: Path) -> None:
    for relative in (
        "LICENSE",
        "CITATION.cff",
        "output/pdf/ripii-manuscript.pdf",
        "output/release/ripii.cdx.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    metadata = {
        "authors": [{"name": "Owner", "affiliation": "Independent"}],
        "copyright_holder": "Owner",
        "software_license_spdx": "MIT",
        "funding": [],
        "conflicts_of_interest": "None declared",
        "author_contributions": "All roles",
        "approved_by": "Owner",
        "approved_at_utc": "2026-09-13T00:00:00Z",
    }
    path = tmp_path / "release/OWNER_METADATA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert check(tmp_path)["status"] == "PASS"
