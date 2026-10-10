"""Prospective alpha-aware planning; retained v1 plans/source stay unchanged."""

from __future__ import annotations

import math

from .power import approximate_paired_seed_count as _legacy_seed_count

FORMAT = "ripii-paired-power-plan-v2"


def exact_sign_flip_resolution_floor(alpha: float) -> int:
    """Smallest nonzero pair count whose two-sided minimum p is <= alpha.

    Two of the 2**n sign assignments always attain the observed absolute sum for
    a nonzero all-same-sign sample. Integer ratios avoid log2 rounding at exact
    powers of two and at neighboring representable significance thresholds.
    This resolution bound is necessary; it is not a power guarantee.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    numerator, denominator = float(alpha).as_integer_ratio()
    pairs = max(1, (2 * denominator).bit_length() - numerator.bit_length())
    while numerator * (1 << pairs) < 2 * denominator:
        pairs += 1
    return pairs


def approximate_paired_seed_count(
    development_differences: list[float],
    *,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    """Retain the v1 normal approximation and correct its exact-test floor.

    This deliberately changes neither the variance estimator nor the power
    approximation. Thresholds too small for v1's normal-tail computation are
    rejected explicitly; the resolution helper itself supports all float alphas.
    """
    floor = exact_sign_flip_resolution_floor(alpha)
    if 1.0 - alpha / 2.0 == 1.0:
        raise ValueError(
            "alpha is below the retained normal approximation's floating-point resolution"
        )
    result = _legacy_seed_count(
        development_differences,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
    )
    result.update(
        format=FORMAT,
        recommended_minimum_pairs=max(floor, result["normal_approximation_pairs"]),
        exact_sign_flip_resolution_floor_pairs=floor,
        exact_sign_flip_resolution_rule="minimum two-sided p = 2/2**n; reject at p <= alpha",
    )
    result["claim_boundary"] += (
        " The resolution floor assumes nonzero pairs and is not a power guarantee."
        " The retained exact enumerator supports at most 20 nonzero pairs; larger"
        " plans require a separately frozen feasible inference method."
    )
    return result
