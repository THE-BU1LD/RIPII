from __future__ import annotations

from pathlib import Path

import pytest

from scripts.render_paper_results import PLACEHOLDER, render_manuscript

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "research/results/development/world_v3_convergence_capsule_v2.json"


def test_paper_result_block_is_capsule_generated(tmp_path: Path) -> None:
    template = tmp_path / "paper.md"
    template.write_text(f"before\n\n{PLACEHOLDER}\n\nafter\n", encoding="utf-8")
    rendered = render_manuscript(template, CAPSULE)
    assert PLACEHOLDER not in rendered
    assert "0.0904" in rendered
    assert "-13.05%" in rendered
    assert "all 5 paired seeds" in rendered


def test_paper_renderer_rejects_missing_or_duplicate_placeholder(
    tmp_path: Path,
) -> None:
    template = tmp_path / "paper.md"
    template.write_text("no placeholder", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        render_manuscript(template, CAPSULE)
    template.write_text(f"{PLACEHOLDER}\n{PLACEHOLDER}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        render_manuscript(template, CAPSULE)
