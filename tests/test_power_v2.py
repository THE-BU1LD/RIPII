from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from ripii.utils.power import approximate_paired_seed_count as legacy_plan
from ripii.utils.power_v2 import (
    approximate_paired_seed_count,
    exact_sign_flip_resolution_floor,
)
from scripts.plan_power import _signature
from scripts.plan_power_v2 import _publish_new, plan, verify_plan_artifact


def minimum_p_by_enumeration(n: int) -> Fraction:
    # All observed differences are +1. Enumerate actual signed sums, independently
    # of the planner's integer-ratio threshold calculation.
    tail = sum(abs(sum(signs)) >= n for signs in itertools.product((-1, 1), repeat=n))
    return Fraction(tail, 2**n)


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        0.05,
        0.01,
        0.001,
        0.0625,
        math.nextafter(0.0625, 0.0),
        math.nextafter(0.0625, 1.0),
    ],
)
def test_resolution_floor_is_minimal_by_independent_sign_enumeration(alpha):
    floor = exact_sign_flip_resolution_floor(alpha)
    threshold = Fraction.from_float(alpha)
    assert minimum_p_by_enumeration(floor) <= threshold
    assert minimum_p_by_enumeration(floor - 1) > threshold


def test_resolution_helper_does_not_underflow_for_smallest_positive_float():
    alpha = math.ulp(0.0)
    floor = exact_sign_flip_resolution_floor(alpha)
    assert floor == 1075
    assert Fraction(2, 2**floor) <= Fraction.from_float(alpha)
    assert Fraction(2, 2 ** (floor - 1)) > Fraction.from_float(alpha)


@pytest.mark.parametrize("alpha, floor", [(0.05, 6), (0.01, 8), (0.001, 11)])
def test_small_variance_plan_respects_its_declared_alpha(alpha, floor):
    result = approximate_paired_seed_count(
        [-0.001, 0.001], minimum_detectable_effect=1.0, alpha=alpha
    )
    assert result["normal_approximation_pairs"] == 1
    assert result["exact_sign_flip_resolution_floor_pairs"] == floor
    assert result["recommended_minimum_pairs"] == floor
    assert result["format"] == "ripii-paired-power-plan-v2"


def test_normal_approximation_can_dominate_resolution_floor_and_keeps_legacy_equations():
    values = [-0.1, -0.05, 0.0, 0.05, 0.1]
    old = legacy_plan(values, minimum_detectable_effect=0.05)
    new = approximate_paired_seed_count(values, minimum_detectable_effect=0.05)
    for key in (
        "development_mean",
        "development_sample_std",
        "normal_approximation_pairs",
        "recommended_minimum_pairs",
        "exact_sign_flip_resolution_floor_pairs",
    ):
        assert new[key] == old[key]
    assert new["recommended_minimum_pairs"] > 6


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan"), float("inf"), True])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        exact_sign_flip_resolution_floor(alpha)


def test_unresolvable_normal_tail_fails_instead_of_claiming_a_plan():
    with pytest.raises(ValueError, match="floating-point resolution"):
        approximate_paired_seed_count(
            [-0.001, 0.001], minimum_detectable_effect=1.0, alpha=1e-30
        )


def write_generated_summary(path: Path) -> Path:
    rows = []
    for seed, delta in enumerate([-0.001, 0.001]):
        for variant, rmse in (("candidate", 1.0 + delta), ("baseline", 1.0)):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "bottleneck": "continuous",
                    "metrics": {"iid": {"position_rmse": rmse}},
                }
            )
    path.write_text(json.dumps({"runs": rows}), encoding="utf-8")
    return path


def create_generated_plan(tmp_path):
    summary = write_generated_summary(tmp_path / "generated.json")
    return plan(
        summary,
        candidate="candidate",
        baseline="baseline",
        split="iid",
        bottleneck="continuous",
        minimum_detectable_effect=1.0,
        alpha=0.01,
        power=0.8,
    )


def test_real_summary_pairing_and_v2_artifact_roundtrip(tmp_path):
    payload = create_generated_plan(tmp_path)
    output = tmp_path / "plan.json"
    _publish_new(output, payload)
    verified = verify_plan_artifact(output)
    assert verified["status"] == "PASS"
    assert verified["recommended_minimum_pairs"] == 8
    assert payload["seeds"] == [0, 1]
    assert set(payload["source_sha256"]) == {
        "scripts/plan_power.py",
        "ripii/utils/power.py",
        "scripts/plan_power_v2.py",
        "ripii/utils/power_v2.py",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(recommended_minimum_pairs=6),
        lambda p: p.update(recommended_minimum_pairs=8.0),
        lambda p: p.update(exact_sign_flip_resolution_floor_pairs=6),
        lambda p: p.update(normal_approximation_pairs=99),
        lambda p: p.update(development_sample_std=1.0),
        lambda p: p.update(seeds=[0, 0]),
        lambda p: p.update(seeds=[0]),
        lambda p: p.update(evidence_status="scientific_complete"),
        lambda p: p["source_sha256"].update({"ripii/utils/power_v2.py": "0" * 64}),
    ],
)
def test_resigned_inconsistent_artifacts_are_rejected(tmp_path, mutation):
    payload = create_generated_plan(tmp_path)
    mutation(payload)
    payload["signature"] = _signature(payload)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_plan_artifact(path)


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_output_is_never_replaced(tmp_path, symlink):
    preserved = tmp_path / "preserved.json"
    preserved.write_bytes(b"original bytes")
    output = tmp_path / "plan.json"
    if symlink:
        output.symlink_to(preserved)
    else:
        output.write_bytes(b"original bytes")
    with pytest.raises(FileExistsError):
        _publish_new(output, {"new": "payload"})
    assert preserved.read_bytes() == b"original bytes"
    assert output.read_bytes() == b"original bytes"
    assert not list(tmp_path.glob(".plan.json.*"))


def test_failed_publication_cleans_only_its_own_staging_file(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise OSError("simulated filesystem error")

    monkeypatch.setattr(os, "link", failure)
    output = tmp_path / "plan.json"
    with pytest.raises(OSError, match="simulated"):
        _publish_new(output, {"valid": True})
    assert not output.exists()
    assert not list(tmp_path.glob(".plan.json.*"))


def test_actual_cli_creates_then_verifies_generated_plan_and_refuses_overwrite(
    tmp_path,
):
    source = write_generated_summary(tmp_path / "generated.json")
    output = tmp_path / "prospective.json"
    command = [
        sys.executable,
        "-m",
        "scripts.plan_power_v2",
        str(source),
        "--candidate",
        "candidate",
        "--baseline",
        "baseline",
        "--split",
        "iid",
        "--minimum-detectable-effect",
        "1",
        "--alpha",
        ".01",
        "--output",
        str(output),
    ]
    created = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(created.stdout)["recommended_minimum_pairs"] == 8
    before = output.read_bytes()
    verified = subprocess.run(
        [sys.executable, "-m", "scripts.plan_power_v2", "--verify-output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(verified.stdout)["status"] == "PASS"
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == before
