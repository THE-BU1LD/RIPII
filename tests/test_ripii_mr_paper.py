from __future__ import annotations

from pathlib import Path


def test_ripii_mr_paper_records_completed_negative_result_and_is_fail_closed() -> None:
    paper = Path("paper/RIPII_MR_MANUSCRIPT.md").read_text(encoding="utf-8")
    assert "matrix completed with a `no_advance` decision" in paper
    assert "none of 60 paired candidate/control" in paper
    assert "makes no positive accuracy, novelty" in paper
    assert "75 training runs" in paper
    assert "no arbitrary direct position correction" in paper
    assert "Failure of the 75-run development gate was the predeclared stop condition" in paper
    assert "This architecture version is therefore terminated" in paper


def test_ripii_mr_protocol_and_entrypoint_are_documented() -> None:
    protocol = Path("research/protocols/ripii_mr_development_v1.md").read_text(
        encoding="utf-8"
    )
    assert "Data seeds: 3101, 3203, 3307" in protocol
    assert "every paired cell" in protocol
    assert "./scripts/start_ripii_mr_overnight.sh" in protocol
