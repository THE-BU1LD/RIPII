from __future__ import annotations

import pytest
import torch
from torch import nn

from ripii.world.experiment import atomic_save, load_model, symmetry_diagnostics
from ripii.world.models import WorldModel
from ripii.world.multiresolution import GeometryCoarsener
from ripii.world.physics import make_dataset


def _force_nonzero_pair_outputs(model: WorldModel) -> None:
    with torch.no_grad():
        model.multiresolution.fine_impulse.net[-1].weight.normal_(0.0, 0.05)
        model.multiresolution.fine_impulse.net[-1].bias.fill_(0.2)
        model.multiresolution.coarse_impulse.net[-1].weight.normal_(0.0, 0.05)
        model.multiresolution.coarse_impulse.net[-1].bias.fill_(0.1)
        model.multiresolution.router[-1].bias.fill_(4.0)


class _FixedRouter(nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits).unsqueeze(-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] != self.logits.shape[0]:
            raise AssertionError("fixed router batch does not match test input")
        return self.logits.to(device=inputs.device, dtype=inputs.dtype)


def test_partition_of_unity_and_mass_restriction() -> None:
    data = make_dataset("test", 4, 2, 103)
    state, mask = data["states"][:, 0], data["mask"]
    hidden = torch.randn(4, 8, 16)
    coarsener = GeometryCoarsener(groups=4, memberships=2)
    coarse = coarsener(
        hidden,
        state[..., :2],
        state[..., 2:4],
        state[..., 5:6],
        mask,
    )
    assert torch.allclose(
        coarse["weights"].sum(-1)[mask],
        torch.ones_like(coarse["weights"].sum(-1)[mask]),
        atol=1e-7,
    )
    fine_mass = (state[..., 5:6] * mask.unsqueeze(-1)).sum(1)
    assert torch.allclose(coarse["mass"].sum(1), fine_mass, atol=1e-6)
    assert (coarse["weights"][~mask] == 0).all()


def test_ripii_mr_is_euclidean_and_permutation_equivariant() -> None:
    data = make_dataset("test", 4, 2, 107)
    state = data["states"][:, 0]
    action, mask = data["actions"][:, 0], data["mask"]
    model = WorldModel("ripii_mr", hidden=16).eval()
    _force_nonzero_pair_outputs(model)
    metrics = symmetry_diagnostics(model, state, action, mask)
    assert metrics["translation_equivariance_rmse"] < 3e-6
    assert metrics["quarter_turn_equivariance_rmse"] < 3e-6

    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 7, 4])
    inverse = permutation.argsort()
    expected = model(state, action, mask)
    permuted = model(
        state[:, permutation], action[:, permutation], mask[:, permutation]
    )[:, inverse]
    assert torch.allclose(permuted, expected, atol=3e-6, rtol=2e-6)


def test_internal_impulses_preserve_momentum_and_angular_momentum() -> None:
    data = make_dataset("test", 6, 2, 109)
    state, mask = data["states"][:, 0], data["mask"]
    action = torch.zeros_like(data["actions"][:, 0])
    model = WorldModel("ripii_mr", hidden=16).eval()
    _force_nonzero_pair_outputs(model)
    after = model(state, action, mask)

    before_momentum = (
        state[..., 2:4] * state[..., 5:6] * mask.unsqueeze(-1)
    ).sum(1)
    after_momentum = (
        after[..., 2:4] * after[..., 5:6] * mask.unsqueeze(-1)
    ).sum(1)
    assert torch.allclose(after_momentum, before_momentum, atol=3e-6, rtol=2e-6)
    diagnostics = model.diagnostics(mask)
    assert diagnostics["internal_momentum_residual_max"] < 3e-6
    assert diagnostics["internal_torque_residual_max"] < 3e-6
    assert diagnostics["partition_unity_max_error"] < 1e-6
    assert diagnostics["restriction_mass_max_error"] < 1e-5


def test_actions_are_applied_as_physical_impulses() -> None:
    data = make_dataset("test", 3, 1, 113)
    state, mask = data["states"][:, 0], data["mask"]
    action = torch.randn_like(data["actions"][:, 0]) * mask.unsqueeze(-1)
    model = WorldModel("ripii_mr", hidden=16).eval()
    after = model(state, action, mask)
    expected_delta_momentum = model.dt * (action * mask.unsqueeze(-1)).sum(1)
    actual_delta_momentum = (
        (after[..., 2:4] - state[..., 2:4])
        * state[..., 5:6]
        * mask.unsqueeze(-1)
    ).sum(1)
    assert torch.allclose(
        actual_delta_momentum, expected_delta_momentum, atol=2e-6, rtol=2e-6
    )


def test_ripii_mr_gradients_reach_fine_coarse_and_router() -> None:
    data = make_dataset("train", 4, 2, 127)
    state, action, mask = (
        data["states"][:, 0],
        data["actions"][:, 0],
        data["mask"],
    )
    model = WorldModel("ripii_mr", hidden=16).train()
    _force_nonzero_pair_outputs(model)
    prediction = model(state, action, mask)
    prediction[..., :4].square().mean().backward()
    for module in (
        model.multiresolution.fine_impulse,
        model.multiresolution.coarse_impulse,
        model.multiresolution.router,
    ):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_ripii_mr_rejects_an_all_padding_scene() -> None:
    model = WorldModel("ripii_mr", hidden=16).eval()
    state = torch.zeros(1, 8, 6)
    action = torch.zeros(1, 8, 2)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one live object"):
        model(state, action, mask)


def test_ripii_mr_v2_all_off_performs_zero_coarse_work() -> None:
    data = make_dataset("test", 3, 1, 131)
    state, action, mask = (
        data["states"][:, 0],
        data["actions"][:, 0],
        data["mask"],
    )
    model = WorldModel("ripii_mr_v2", hidden=16).eval()
    model.multiresolution.router = _FixedRouter([-100.0] * len(state))
    calls = []
    model.multiresolution.coarsener.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape[0])
    )

    result = model(state, action, mask)
    diagnostics = model.diagnostics(mask)

    assert torch.isfinite(result).all()
    assert calls == []
    assert diagnostics["routing_rate"] == 0
    assert diagnostics["coarse_compute_calls"] == 0
    assert diagnostics["coarse_scenes_processed"] == 0
    assert model.multiresolution.compute_counters() == {
        "coarse_compute_calls": 0,
        "coarse_scenes_processed": 0,
    }


def test_ripii_mr_v2_all_on_executes_one_coarse_subbatch() -> None:
    data = make_dataset("test", 3, 1, 137)
    state, action, mask = (
        data["states"][:, 0],
        data["actions"][:, 0],
        data["mask"],
    )
    model = WorldModel("ripii_mr_v2", hidden=16).eval()
    model.multiresolution.router = _FixedRouter([100.0] * len(state))
    calls = []
    model.multiresolution.coarsener.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape[0])
    )

    model(state, action, mask)
    diagnostics = model.diagnostics(mask)

    assert calls == [len(state)]
    assert diagnostics["routing_rate"] == 1
    assert diagnostics["coarse_compute_calls"] == 1
    assert diagnostics["coarse_scenes_processed"] == len(state)
    assert model.multiresolution.compute_counters() == {
        "coarse_compute_calls": 1,
        "coarse_scenes_processed": len(state),
    }


def test_ripii_mr_v2_mixed_routing_only_processes_selected_scenes() -> None:
    data = make_dataset("test", 4, 1, 139)
    state, action, mask = (
        data["states"][:, 0],
        data["actions"][:, 0],
        data["mask"],
    )
    model = WorldModel("ripii_mr_v2", hidden=16).eval()
    model.multiresolution.router = _FixedRouter([-100.0, 100.0, -100.0, 100.0])
    calls = []
    model.multiresolution.coarsener.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape[0])
    )

    model(state, action, mask)
    diagnostics = model.diagnostics(mask)

    assert calls == [2]
    assert diagnostics["routing_rate"] == 0.5
    assert diagnostics["routed_scene_count"] == 2
    assert diagnostics["coarse_scenes_processed"] == 2
    assert (model.last_assignments[[0, 2]] == 0).all()
    routed_mask = mask[[1, 3]]
    routed_assignments = model.last_assignments[[1, 3]]
    assert torch.allclose(
        routed_assignments.sum(-1)[routed_mask],
        torch.ones_like(routed_assignments.sum(-1)[routed_mask]),
        atol=1e-7,
    )


def test_ripii_mr_v2_profiler_counters_reset_and_checkpoint_roundtrip(
    tmp_path,
) -> None:
    data = make_dataset("test", 2, 1, 149)
    state, action, mask = (
        data["states"][:, 0],
        data["actions"][:, 0],
        data["mask"],
    )
    model = WorldModel("ripii_mr_v2", hidden=16).eval()
    model.multiresolution.router = _FixedRouter([100.0, -100.0])
    model(state, action, mask)
    model(state, action, mask)
    assert model.multiresolution.compute_counters() == {
        "coarse_compute_calls": 2,
        "coarse_scenes_processed": 2,
    }
    model.multiresolution.reset_compute_counters()
    assert model.multiresolution.compute_counters() == {
        "coarse_compute_calls": 0,
        "coarse_scenes_processed": 0,
    }

    # Checkpoint the registered architecture, not the test-only fixed router.
    checkpoint_model = WorldModel("ripii_mr_v2", hidden=16).eval()
    path = tmp_path / "ripii_mr_v2.pt"
    atomic_save(
        path,
        {
            "format": "ripii-world-v1",
            "model": checkpoint_model.state_dict(),
            "model_spec": checkpoint_model.spec,
            "completed_steps": 1,
            "best_validation": 0.5,
        },
    )
    loaded, checkpoint = load_model(path)
    assert loaded.variant == "ripii_mr_v2"
    assert checkpoint["model_spec"] == checkpoint_model.spec
