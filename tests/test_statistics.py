from __future__ import annotations

import math

import pytest

from ripii.utils.statistics import (
    exact_sign_flip_pvalue,
    holm_adjust,
    paired_summary,
    practical_equivalence_summary,
)


def test_paired_statistics_use_pairs_and_are_deterministic() -> None:
    result = paired_summary([2, 4, 6], [1, 2, 3], bootstrap_seed=7)
    assert result["differences"] == [1.0, 2.0, 3.0]
    assert result["mean_difference"] == 2.0
    assert result == paired_summary([2, 4, 6], [1, 2, 3], bootstrap_seed=7)
    assert exact_sign_flip_pvalue([1, 1, 1]) == 0.25


def test_statistics_fail_closed_and_holm_is_monotone() -> None:
    with pytest.raises(ValueError):
        paired_summary([1], [1, 2])
    with pytest.raises(FloatingPointError):
        paired_summary([math.nan], [1])
    with pytest.raises(ValueError, match="baseline mean"):
        paired_summary([1, -1], [1, -1])
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_practical_equivalence_uses_paired_relative_margin() -> None:
    equivalent = practical_equivalence_summary(
        [1.01, 0.98, 1.02], [1.0, 1.0, 1.0], bootstrap_seed=7
    )
    assert equivalent["descriptive_decision"] == "within_margin_for_executed_seeds"
    assert equivalent["bootstrap_interval_within_margin"] is True
    outside = practical_equivalence_summary([1.1, 1.0], [1.0, 1.0])
    assert outside["descriptive_decision"] == "not_within_margin"
    with pytest.raises(ValueError, match="baselines positive"):
        practical_equivalence_summary([1.0], [0.0])
