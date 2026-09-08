from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ripii.world.experiment import (
    CAPACITY_MATCH_TOLERANCE,
    Experiment,
    ForceKinematic,
    Kinematic,
    Persistence,
    benchmark,
    capture,
    load_model,
    train,
    verify,
    verify_capsule,
    widths,
)
from ripii.world.models import VARIANTS, WorldModel
from ripii.world.physics import Physics, make_dataset, sample_scene, simulate


def test_physics_identity_motion_and_pair_momentum():
    state = torch.tensor(
        [[[-0.09, 0.0, 0.3, 0.0, 0.1, 0.7], [0.09, 0.0, -0.2, 0.0, 0.1, 1.4]]]
    )
    mask = torch.ones(1, 2, dtype=torch.bool)
    after = simulate(state, torch.zeros(1, 2, 2), mask, Physics(drag=0))
    before_momentum = (state[..., 2:4] * state[..., 5:6]).sum(1)
    after_momentum = (after[..., 2:4] * after[..., 5:6]).sum(1)
    assert torch.allclose(before_momentum, after_momentum, atol=1e-6)
    assert after[0, 0, 2] < state[0, 0, 2]
    assert after[0, 1, 2] > state[0, 1, 2]
    still = state.clone()
    still[..., :2] *= 3
    still[..., 2:4] = 0
    assert torch.equal(simulate(still, torch.zeros(1, 2, 2), mask), still)


def test_walls_actions_and_masked_objects():
    state = torch.tensor(
        [[[0.98, 0.0, 0.3, 0.0, 0.1, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    mask = torch.tensor([[True, False]])
    actions = torch.tensor([[[0.0, 1.0], [999.0, 999.0]]])
    after = simulate(state, actions, mask)
    assert after[0, 0, 2] < state[0, 0, 2]
    assert after[0, 0, 3] > 0
    assert torch.equal(after[0, 1], torch.zeros(6))


def test_world_paths_fail_closed_on_invalid_numerics():
    with pytest.raises(ValueError):
        Physics(dt=float("nan"))
    state, mask = sample_scene(torch.Generator().manual_seed(1), 2)
    action = torch.zeros(1, 8, 2)
    bad_state = state[None].clone()
    bad_state[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        simulate(bad_state, action, mask[None])
    model = WorldModel("graph", hidden=16)
    with pytest.raises(FloatingPointError):
        model(bad_state, action, mask[None])
    with pytest.raises(ValueError):
        WorldModel("multiscale", hidden=16, groups=9)


def test_analytic_baselines_have_distinct_control_contracts():
    state, mask = sample_scene(torch.Generator().manual_seed(4), 2)
    state = state[None]
    mask = mask[None]
    action = torch.zeros(1, 8, 2)
    action[:, 0] = torch.tensor([1.0, -0.5])
    persistent = Persistence()(state, action, mask)
    constant_velocity = Kinematic()(state, action, mask)
    force_kinematic = ForceKinematic()(state, action, mask)
    assert torch.equal(persistent, state)
    assert not torch.equal(constant_velocity[..., :2], persistent[..., :2])
    assert not torch.equal(force_kinematic[..., 2:4], constant_velocity[..., 2:4])


def test_default_world_models_satisfy_capacity_gate():
    cfg = Experiment()
    for bottleneck in ("continuous", "fsq", "vq"):
        matches = widths(cfg, list(VARIANTS), bottleneck)
        assert all(
            details["relative_error"] <= CAPACITY_MATCH_TOLERANCE
            for details in matches.values()
        )


def test_benchmark_rejects_unmatched_capacity_before_writing(tmp_path: Path):
    output = tmp_path / "unmatched"
    with pytest.raises(ValueError, match="capacity matching exceeds"):
        benchmark(
            Experiment(hidden=8),
            output,
            seeds=[3],
            variants=["graph", "multiscale"],
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("seeds", "variants", "bottlenecks"),
    [
        ([True], ["graph"], ["continuous"]),
        ([-1], ["graph"], ["continuous"]),
        ([3], [], ["continuous"]),
        ([3], ["unknown"], ["continuous"]),
        ([3], ["graph"], []),
        ([3], ["graph"], ["unknown"]),
        ([3], ["graph"], ["continuous", "continuous"]),
    ],
)
def test_benchmark_rejects_invalid_grid_before_writing(
    tmp_path: Path, seeds, variants, bottlenecks
):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="valid, nonempty, and unique"):
        benchmark(
            Experiment(hidden=16),
            output,
            seeds=seeds,
            variants=variants,
            bottlenecks=bottlenecks,
        )
    assert not output.exists()


def test_split_reproducibility_and_actual_heldout_properties():
    a = make_dataset("train", 8, 4, 7)
    b = make_dataset("train", 8, 4, 7)
    test = make_dataset("test", 8, 4, 7)
    composed = make_dataset("composition", 8, 4, 7)
    larger = make_dataset("more_objects", 8, 4, 7)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not set(a["ids"].tolist()) & set(test["ids"].tolist())
    properties = a["states"][:, 0, :, 4:][a["mask"]]
    assert not ((properties[:, 0] > 0.1) & (properties[:, 1] > 1)).any()
    assert (composed["states"][:, 0, :, 5][composed["mask"]] > 1).all()
    assert (larger["mask"].sum(1) > 4).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("bottleneck", ["continuous", "fsq", "vq"])
def test_all_models_learn_and_preserve_properties(variant, bottleneck):
    torch.set_num_threads(1)
    data = make_dataset("train", 4, 2, 9)
    state, action, mask = data["states"][:, 0], data["actions"][:, 0], data["mask"]
    model = WorldModel(variant, hidden=16, bottleneck=bottleneck)
    output = model(state, action, mask)
    loss = (
        output[..., :4] - data["states"][:, 1, :, :4]
    ).square().mean() + 0.01 * model.aux_loss
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert torch.equal(output[..., 4:], state[..., 4:])
    assert (output[~mask] == 0).all()
    diagnostics = model.diagnostics(mask)
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
    if variant == "multiscale":
        assert 1 <= diagnostics["assignment_effective_groups"] <= 4
        assert 0 <= diagnostics["assignment_normalized_entropy"] <= 1
    if bottleneck == "fsq":
        assert 0 < diagnostics["fsq_level_utilization"] <= 1
        assert 1 <= diagnostics["fsq_effective_levels"] <= 5
    if bottleneck == "vq":
        assert 0 < diagnostics["vq_coarse_usage"] <= 1
        assert 0 < diagnostics["vq_fine_usage"] <= 1


@pytest.mark.parametrize(
    "variant", ["graph", "transformer", "global_pool", "multiscale"]
)
def test_learned_interactions_are_permutation_equivariant(variant):
    torch.set_num_threads(1)
    data = make_dataset("train", 3, 2, 9)
    state, action, mask = data["states"][:, 0], data["actions"][:, 0], data["mask"]
    model = WorldModel(variant, hidden=16).eval()
    # Nonzero head ensures the test exercises messages rather than only the prior.
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)
    order = torch.tensor([4, 2, 0, 7, 1, 3, 6, 5])
    with torch.no_grad():
        expected = model(state, action, mask)[:, order]
        actual = model(state[:, order], action[:, order], mask[:, order])
    assert torch.allclose(actual, expected, atol=2e-6)


def test_training_resume_is_exact(tmp_path: Path):
    torch.set_num_threads(1)
    cfg = Experiment(
        steps=12,
        hidden=16,
        train_scenes=12,
        eval_scenes=4,
        train_horizon=4,
        test_horizon=4,
        batch_size=4,
        rollout_steps=2,
        validate_every=2,
    )
    train(cfg, tmp_path / "full", seed=3)
    train(replace(cfg, steps=6), tmp_path / "split", seed=3)
    train(cfg, tmp_path / "split", seed=3, resume=tmp_path / "split/latest.pt")
    full, _ = load_model(tmp_path / "full/final.pt")
    split, _ = load_model(tmp_path / "split/final.pt")
    assert all(
        torch.equal(value, split.state_dict()[key])
        for key, value in full.state_dict().items()
    )
    with pytest.raises(ValueError, match="already complete"):
        train(cfg, tmp_path / "split", seed=3, resume=tmp_path / "split/final.pt")


def test_benchmark_verification_and_demo_interventions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch.set_num_threads(1)
    cfg = Experiment(
        steps=2,
        hidden=28,
        train_scenes=4,
        eval_scenes=2,
        train_horizon=2,
        test_horizon=4,
        batch_size=2,
        rollout_steps=2,
        validate_every=1,
    )
    output = tmp_path / "benchmark"
    result = benchmark(cfg, output, seeds=[3], variants=["graph", "multiscale"])
    assert len(result["runs"]) == 2
    assert result["decision"] == "no_advance"
    assert verify(output)["status"] == "PASS"
    monkeypatch.chdir(tmp_path)
    assert verify(Path("benchmark"))["status"] == "PASS"
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["capacity_match_tolerance"] == CAPACITY_MATCH_TOLERANCE
    assert "ripii/models/quantizer.py" in protocol["source_sha256"]
    assert "pyproject.toml" in protocol["source_sha256"]
    assert "uv.lock" in protocol["source_sha256"]
    assert (output / "source/ripii/models/quantizer.py").is_file()
    assert set(result["analytic_baselines"]) == {
        "persistence",
        "constant_velocity",
        "force_kinematic",
    }
    assert set(result["paired_controls"]) == {"graph"}
    graph_control = result["paired_controls"]["graph"]
    assert graph_control["seeds"] == 1
    assert len(graph_control["pairs"]) == 1
    assert graph_control["claim_boundary"].startswith("Descriptive")
    multiscale_metrics = result["runs"][1]["metrics"]["test"]
    assert "assignment_effective_groups" in multiscale_metrics
    assert "assignment_temporal_change" in multiscale_metrics
    from ripii.world.demo import main

    demo = main(output / "multiscale_continuous/seed_3/best.pt", tmp_path / "demo.png")
    original = demo.truth.clone()
    old_prediction = demo.prediction.clone()
    demo.force = torch.tensor([1.0, 0.0])
    demo.recompute()
    assert not torch.equal(demo.truth, original)
    assert not torch.equal(demo.prediction, old_prediction)
    from types import SimpleNamespace

    demo.rewind()
    start = demo.initial[0, :2].clone()
    demo.press(
        SimpleNamespace(
            inaxes=demo.axes[0], xdata=float(start[0]), ydata=float(start[1]), button=3
        )
    )
    demo.release(
        SimpleNamespace(
            inaxes=demo.axes[0],
            xdata=float(start[0] + 0.1),
            ydata=float(start[1] + 0.1),
        )
    )
    assert torch.allclose(demo.initial[0, 2:4], torch.tensor([0.2, 0.2]), atol=1e-6)
    demo.change_objects(6)
    assert int(demo.mask.sum()) == 6
    before_palette = demo.prediction.clone()
    demo.change_palette("Recolored")
    assert torch.equal(before_palette, demo.prediction)
    assert (tmp_path / "demo.png").stat().st_size > 10000

    capsule_path = tmp_path / "capsule.json"
    capsule = capture(output, capsule_path)
    assert capsule["decision"] == "no_advance"
    verified_capsule = verify_capsule(capsule_path)
    assert verified_capsule["status"] == "PASS"
    assert verified_capsule["retained_files"] == 4
    assert verified_capsule["content_hashes_verified"] is True
    with pytest.raises(FileExistsError, match="refusing existing"):
        capture(output, capsule_path)
    tampered = json.loads(capsule_path.read_text(encoding="utf-8"))
    tampered["decision"] = "advance"
    capsule_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_capsule(capsule_path)
    extra = output / "undeclared.txt"
    extra.write_text("not in manifest", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        verify(output)
    extra.unlink()
    (output / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="verification failed"):
        verify(output)


def test_scene_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        sample_scene(torch.Generator(), 9, 8)
