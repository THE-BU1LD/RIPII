from __future__ import annotations

from scripts.qualify_quantizer import run_qualification


def test_quantizer_qualification_is_deterministic_and_bounded() -> None:
    kwargs = {
        "seed": 13,
        "steps": 4,
        "samples": 64,
        "code_dim": 8,
        "coarse_codes": 4,
        "fine_codes": 4,
        "learning_rate": 0.02,
    }
    first = run_qualification(**kwargs)
    second = run_qualification(**kwargs)
    assert first == second
    assert first["evidence_status"] == "development_only"
    assert first["qualified"] == all(first["gates"].values())
    for name, value in first["metrics"].items():
        assert value >= 0.0, name
