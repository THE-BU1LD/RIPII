from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence


def _finite(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise FloatingPointError(f"{name} contains a non-finite value")
    return result


def paired_differences(
    candidate: Sequence[float], baseline: Sequence[float]
) -> list[float]:
    left, right = _finite(candidate, "candidate"), _finite(baseline, "baseline")
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    return [a - b for a, b in zip(left, right)]


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided paired randomization p-value for the mean difference.

    Initialization seeds are the experimental units. Zero differences are removed;
    exact enumeration is deliberately limited so this cannot silently become an
    exponential-time analysis.
    """
    values = [value for value in _finite(differences, "differences") if value != 0.0]
    if not values:
        return 1.0
    if len(values) > 20:
        raise ValueError(
            "exact sign-flip enumeration supports at most 20 nonzero pairs"
        )
    observed = abs(sum(values) / len(values))
    extreme = 0
    total = 2 ** len(values)
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(
            sum(sign * value for sign, value in zip(signs, values)) / len(values)
        )
        extreme += statistic + tolerance >= observed
    return extreme / total


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    draws: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    sample = _finite(values, "values")
    if not 0.0 < confidence < 1.0 or draws < 100:
        raise ValueError("confidence must be in (0, 1) and draws must be at least 100")
    rng = random.Random(seed)
    n = len(sample)
    means = sorted(
        sum(sample[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, math.floor(alpha * draws))]
    hi = means[min(draws - 1, math.ceil((1.0 - alpha) * draws) - 1)]
    return lo, hi


def paired_summary(
    candidate: Sequence[float], baseline: Sequence[float], *, bootstrap_seed: int = 0
) -> dict[str, float | int | list[float]]:
    differences = paired_differences(candidate, baseline)
    n = len(differences)
    mean = sum(differences) / n
    median = (
        sorted(differences)[n // 2]
        if n % 2
        else sum(sorted(differences)[n // 2 - 1 : n // 2 + 1]) / 2
    )
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in differences) / (n - 1))
        if n > 1
        else 0.0
    )
    ci = bootstrap_mean_ci(differences, seed=bootstrap_seed)
    baseline_values = _finite(baseline, "baseline")
    scale = sum(baseline_values) / len(baseline_values)
    if scale == 0.0:
        raise ValueError("baseline mean must be nonzero for relative difference")
    return {
        "n_pairs": n,
        "differences": differences,
        "mean_difference": mean,
        "median_difference": median,
        "sample_std_difference": std,
        "bootstrap_95_ci_mean_difference": [ci[0], ci[1]],
        "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
        "relative_mean_difference": mean / scale,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = _finite(p_values, "p_values")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def descriptive_summary(values: Sequence[float]) -> dict[str, float | int]:
    sample = _finite(values, "values")
    n = len(sample)
    mean = sum(sample) / n
    ordered = sorted(sample)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in sample) / (n - 1))
        if n > 1
        else 0.0
    )
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "sample_std": std,
        "min": ordered[0],
        "max": ordered[-1],
    }


def paired_sign_flip_test(
    candidate: Sequence[float], baseline: Sequence[float]
) -> dict[str, float | int]:
    differences = paired_differences(candidate, baseline)
    return {
        "n_pairs": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "two_sided_p": exact_sign_flip_pvalue(differences),
    }
