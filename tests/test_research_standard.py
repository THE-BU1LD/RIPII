from __future__ import annotations

import re
from pathlib import Path


def test_final_research_audit_contains_required_decision_fields() -> None:
    audit = Path("FINAL_RESEARCH_AUDIT.md").read_text(encoding="utf-8")
    required = (
        "## Scientific Question",
        "## Main Hypothesis",
        "## Main Contribution",
        "## Strongest Evidence",
        "## Strongest Baseline",
        "## Main Quantitative Result",
        "## Most Important Ablation",
        "## Strongest Robustness Result",
        "## Most Important Failure",
        "## Primary Limitation",
        "## Mathematical Risk",
        "## Experimental Risk",
        "## Reproducibility Risk",
        "## Reviewer Attack Surface",
        "## Unresolved Issues",
        "## Score",
        "## Readiness Level",
        "## Recommendation",
    )
    assert all(heading in audit for heading in required)
    score = re.search(r"\*\*Score: (\d+)/100", audit)
    assert score is not None
    assert int(score.group(1)) < 85
    assert "**P2 — repaired-method evaluation complete" in audit


def test_paper_standard_requires_all_submission_gates() -> None:
    standard = Path("PAPER_STANDARD.md").read_text(encoding="utf-8")
    assert "Only P6 may be described as submission-ready" in standard
    assert "score >=85" in standard
    assert "zero critical failures" in standard
    assert "zero blocking scientific" in standard
