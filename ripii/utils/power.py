from __future__ import annotations

import math
from statistics import NormalDist


def approximate_paired_seed_count(
    development_differences: list[float],
    *,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    """Approximate a future paired design from development-seed variance.

    The normal approximation is planning support, not a confirmatory test or a reason
    to alter the effect threshold after future outcomes are observed.
    """
    values = [float(value) for value in development_differences]
    if (
        len(values) < 2
        or not all(math.isfinite(value) for value in values)
        or not math.isfinite(minimum_detectable_effect)
        or minimum_detectable_effect <= 0
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
        or not math.isfinite(power)
        or not 0 < power < 1
    ):
        raise ValueError("invalid paired power-planning inputs")
    mean = sum(values) / len(values)
    sample_std = math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    if sample_std == 0:
        raise ValueError("development variance is zero; seed count is not estimable")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    approximate = math.ceil(
        ((z_alpha + z_power) * sample_std / minimum_detectable_effect) ** 2
    )
    # Two-sided exact sign-flip inference cannot attain p < .05 below six
    # nonzero pairs (minimum p = 2 / 2**n).
    recommended = max(6, approximate)
    return {
        "format": "ripii-paired-power-plan-v1",
        "development_pairs": len(values),
        "development_differences": values,
        "development_mean": mean,
        "development_sample_std": sample_std,
        "minimum_detectable_effect": minimum_detectable_effect,
        "two_sided_alpha": alpha,
        "target_power": power,
        "normal_approximation_pairs": approximate,
        "recommended_minimum_pairs": recommended,
        "exact_sign_flip_resolution_floor_pairs": 6,
        "method": "paired normal approximation using development-seed sample SD",
        "claim_boundary": (
            "Planning estimate only; freeze the final count and effect threshold before "
            "external test evaluation, and revise upward if external pilot variance is larger."
        ),
    }
