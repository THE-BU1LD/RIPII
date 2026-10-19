from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from ripii.world.closure import (
    CentralPairForce,
    ClosureCoarsener,
    ClosureConfig,
    ClosureModel,
    load_closure_checkpoint,
    project_internal_wrench,
)
from ripii.world.physics import Physics, simulate

KINDS = ("local", "global", "hierarchy", "mean", "law", "prior")


def scene(dtype: torch.dtype = torch.float64):
    state = torch.zeros(2, 8, 6, dtype=dtype)
    state[0, :5] = torch.tensor(
        [
            [-0.55, 0.07, 0.21, -0.13, 0.07, 0.7],
            [0.18, -0.26, -0.14, 0.09, 0.11, 1.4],
            [0.49, 0.48, 0.08, 0.17, 0.09, 1.1],
            [-0.28, -0.48, 0.11, 0.07, 0.08, 0.9],
            [0.72, -0.13, -0.19, -0.06, 0.1, 1.7],
        ],
        dtype=dtype,
    )
    state[1, :3] = torch.tensor(
        [
            [0.93, 0.06, 0.35, -0.08, 0.1, 0.7],
            [0.19, 0.21, -0.06, 0.03, 0.07, 1.4],
            [0.32, 0.26, 0.07, -0.02, 0.09, 0.9],
        ],
        dtype=dtype,
    )
    mask = state[..., 5] > 0
    action = torch.tensor(
        [[0.12, -0.08], [-0.07, 0.13], [0.05, 0.03], [0.04, -0.11], [-0.02, 0.06],
         [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=dtype,
    ).expand(2, -1, -1).clone().masked_fill(~mask.unsqueeze(-1), 0)
    return state, action, mask


def model(kind: str, **kwargs) -> ClosureModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        result = ClosureModel(ClosureConfig(kind=kind, **kwargs)).double()
    generator = torch.Generator().manual_seed(321)
    with torch.no_grad():
        for module in result.modules():
            if isinstance(module, CentralPairForce):
                final = module.net[-1]
                assert isinstance(final, nn.Linear)
                final.weight.uniform_(-0.08, 0.08, generator=generator)
                final.bias.fill_(0.013)
    return result


def torque(position: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    return (position[..., 0] * force[..., 1] - position[..., 1] * force[..., 0]).sum(1)


def momentum(state: torch.Tensor) -> torch.Tensor:
    return (state[..., 5:6] * state[..., 2:4]).sum(1)


def angular_momentum(state: torch.Tensor) -> torch.Tensor:
    return torque(state[..., :2], state[..., 5:6] * state[..., 2:4])


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "oracle"}, {"kind": []}, {"closure": 1}, {"substeps": 0},
        {"substeps": True}, {"hidden": 0}, {"groups": 9}, {"memberships": 0},
        {"memberships": 5}, {"groups": 2.0}, {"local_radius": 0},
        {"force_scale": -1}, {"dt": -0.05}, {"drag": -0.06},
        {"stiffness": -1}, {"damping": -1}, {"learned_coupling_initial": 0},
        {"learned_coupling_initial": -0.25}, {"dt": math.nan},
        {"force_scale": math.inf}, {"damping": "1.5"}, {"drag": False},
    ],
)
def test_invalid_config(change):
    with pytest.raises((TypeError, ValueError)):
        ClosureConfig(**change)


def test_config_is_frozen_and_complete():
    config = ClosureConfig()
    assert ClosureConfig(**asdict(config)) == config
    with pytest.raises(FrozenInstanceError):
        config.kind = "global"
    with pytest.raises(TypeError, match="ClosureConfig"):
        ClosureModel(asdict(config))


def test_exact_parameter_control_and_initialization():
    expected_counts = {"local": 1521, "global": 2386, "hierarchy": 2386,
                       "mean": 1522, "law": 1, "prior": 0}
    arms = {}
    for kind in KINDS:
        torch.manual_seed(997)
        arms[kind] = ClosureModel(ClosureConfig(kind=kind))
        assert sum(p.numel() for p in arms[kind].parameters()) == expected_counts[kind]
    assert [p.shape for p in arms["global"].parameters()] == [
        p.shape for p in arms["hierarchy"].parameters()
    ]
    for kind in ("global", "hierarchy", "mean"):
        for name, parameter in arms["local"].named_parameters():
            assert torch.equal(parameter, dict(arms[kind].named_parameters())[name])
    for kind in ("local", "global", "hierarchy", "mean"):
        for module in arms[kind].modules():
            if isinstance(module, CentralPairForce):
                assert torch.count_nonzero(module.net[-1].weight) == 0
                assert torch.count_nonzero(module.net[-1].bias) == 0
    assert arms["local"].second is None and arms["local"].coarsener is None
    assert arms["global"].coarsener is None
    assert arms["hierarchy"].coarsener is not None
    for kind in KINDS:
        coupling = arms[kind].learned_coupling()
        if kind in ("mean", "law"):
            assert coupling is not None
            torch.testing.assert_close(coupling, torch.tensor(0.25))
        else:
            assert coupling is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_padding_is_sanitized_before_arithmetic(kind, dtype):
    state, action, mask = scene(dtype)
    predictor = model(kind).to(dtype=dtype)
    dirty_state = state.masked_fill(~mask.unsqueeze(-1), torch.finfo(dtype).max / 4)
    dirty_action = action.masked_fill(~mask.unsqueeze(-1), -torch.finfo(dtype).max / 4)
    dirty_state.requires_grad_()
    dirty_action.requires_grad_()
    result = predictor(dirty_state, dirty_action, mask)
    assert torch.equal(result, predictor(state, action, mask))
    assert torch.count_nonzero(result[~mask]) == 0
    assert torch.equal(result[..., 4:6][mask], state[..., 4:6][mask])
    result[..., :4].square().sum().backward()
    assert torch.isfinite(dirty_state.grad).all()
    assert torch.isfinite(dirty_action.grad).all()
    assert torch.count_nonzero(dirty_state.grad[~mask]) == 0
    assert torch.count_nonzero(dirty_action.grad[~mask]) == 0
    for parameter in predictor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("method", ["forward", "internal_force", "external_force"])
@pytest.mark.parametrize(
    "case, error",
    [
        ("state_shape", ValueError), ("action_shape", ValueError),
        ("mask_shape", ValueError), ("mask_dtype", TypeError),
        ("empty_scene", ValueError), ("empty_batch", ValueError),
        ("too_many_slots", ValueError), ("integer_state", TypeError),
        ("mixed_dtype", TypeError), ("low_precision", TypeError),
        ("nonfinite_live", FloatingPointError), ("nonfinite_padding", FloatingPointError),
        ("nonfinite_action", FloatingPointError), ("negative_mass", ValueError),
        ("zero_radius", ValueError),
    ],
)
def test_input_contract_all_public_force_methods(method, case, error):
    state, action, mask = scene()
    if case == "state_shape":
        state = state[..., :5]
    elif case == "action_shape":
        action = action[..., :1]
    elif case == "mask_shape":
        mask = mask[:1]
    elif case == "mask_dtype":
        mask = mask.double()
    elif case == "empty_scene":
        mask[1] = False
    elif case == "empty_batch":
        state, action, mask = state[:0], action[:0], mask[:0]
    elif case == "too_many_slots":
        state = torch.cat((state, state[:, :1]), 1)
        action = torch.cat((action, action[:, :1]), 1)
        mask = torch.cat((mask, mask[:, :1]), 1)
    elif case == "integer_state":
        state = state.long()
    elif case == "mixed_dtype":
        action = action.float()
    elif case == "low_precision":
        state, action = state.half(), action.half()
    elif case == "nonfinite_live":
        state[0, 0, 0] = math.nan
    elif case == "nonfinite_padding":
        state[0, -1, 0] = math.inf
    elif case == "nonfinite_action":
        action[0, -1, 0] = math.nan
    elif case == "negative_mass":
        state[0, 0, 5] = -1
    elif case == "zero_radius":
        state[0, 0, 4] = 0
    with pytest.raises(error):
        getattr(model("hierarchy"), method)(state, action, mask)


def test_rollout_contract_is_checked_before_any_step(monkeypatch):
    state, action, mask = scene()
    predictor = model("hierarchy")

    def fail_if_called(*args):
        pytest.fail("step executed before the complete request was validated")

    monkeypatch.setattr(predictor, "_step", fail_if_called)
    empty = torch.empty(2, 0, 8, 2, dtype=state.dtype)
    actual = predictor.rollout(state, empty, mask)
    assert torch.equal(actual, state.unsqueeze(1))
    with pytest.raises(TypeError, match="boolean"):
        predictor.rollout(state, empty, mask.double())
    invalid_state = state.clone()
    invalid_state[0, 0, 5] = 0
    with pytest.raises(ValueError, match="positive"):
        predictor.rollout(invalid_state, empty, mask)
    actions = action[:, None].expand(-1, 3, -1, -1).clone()
    actions[0, -1, 0, 0] = math.nan
    with pytest.raises(FloatingPointError):
        predictor.rollout(state, actions, mask)
    with pytest.raises(ValueError, match="actions"):
        predictor.rollout(state, empty[:, :, :, :1], mask)
    with pytest.raises(TypeError, match="dtype"):
        predictor.rollout(state, empty.float(), mask)


def test_rollout_includes_initial_state_and_keeps_action_gradients():
    state = torch.tensor([[[0.0, 0.0, 0.1, -0.2, 0.1, 0.05]]], dtype=torch.float64)
    actions = torch.tensor([[[[0.2, -0.3]], [[-0.4, 0.5]], [[0.6, 0.7]]]],
                           dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 1, dtype=torch.bool)
    predictor = model("prior", closure=False)
    trajectory = predictor.rollout(state, actions, mask)
    assert trajectory.shape == (1, 4, 1, 6)
    assert torch.equal(trajectory[:, 0], state)
    assert torch.equal(trajectory[..., 4:6], state[:, None, :, 4:6].expand(-1, 4, -1, -1))
    trajectory[:, -1, :, 2:4].sum().backward()
    torch.testing.assert_close(actions.grad, torch.ones_like(actions))


def test_model_dtype_mismatch_is_explicit():
    state, action, mask = scene()
    with pytest.raises(TypeError, match="model parameters"):
        ClosureModel()(state, action, mask)


@pytest.mark.parametrize("kind", KINDS)
def test_internal_force_and_torque_are_zero_sum(kind):
    state, action, mask = scene()
    force = model(kind).internal_force(state, action, mask)
    torch.testing.assert_close(force.sum(1), torch.zeros(2, 2, dtype=state.dtype),
                               atol=2e-12, rtol=0)
    torch.testing.assert_close(torque(state[..., :2], force), torch.zeros(2, dtype=state.dtype),
                               atol=2e-12, rtol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_generic_internal_e2_and_permutation_equivariance(kind):
    state, action, mask = scene()
    predictor = model(kind, closure=False)
    angle = 0.371
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]], dtype=state.dtype)
    reflection = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=state.dtype)
    matrix = rotation @ reflection
    translated = state.clone()
    translated[..., :2] = state[..., :2] @ matrix.T + torch.tensor([0.31, -0.27])
    translated[..., 2:4] = state[..., 2:4] @ matrix.T
    transformed_action = action @ matrix.T
    permutation = torch.tensor([6, 2, 4, 7, 1, 0, 5, 3])
    expected = predictor.internal_force(state, action, mask)[:, permutation] @ matrix.T
    actual = predictor.internal_force(translated[:, permutation],
                                      transformed_action[:, permutation], mask[:, permutation])
    torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reflection", [False, True])
def test_full_square_environment_symmetry(kind, reflection):
    state, action, mask = scene()
    matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=state.dtype)
    if reflection:
        matrix = matrix @ torch.diag(torch.tensor([-1.0, 1.0], dtype=state.dtype))
    transformed = state.clone()
    transformed[..., :2] = state[..., :2] @ matrix.T
    transformed[..., 2:4] = state[..., 2:4] @ matrix.T
    predictor = model(kind)
    expected = predictor(state, action, mask)
    expected[..., :2] = expected[..., :2] @ matrix.T
    expected[..., 2:4] = expected[..., 2:4] @ matrix.T
    actual = predictor(transformed, action @ matrix.T, mask)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("closure", [False, True])
def test_substep_external_linear_and_angular_impulse_balance(kind, closure, monkeypatch):
    state, action, mask = scene()
    predictor = model(kind, closure=closure)
    calls = []
    original = predictor._external_force

    def capture(current, current_action):
        external = original(current, current_action)
        calls.append((current[..., :2].detach().clone(), external.detach().clone()))
        return external

    monkeypatch.setattr(predictor, "_external_force", capture)
    result = predictor(state, action, mask)
    assert len(calls) == predictor.config.substeps
    assert not torch.equal(calls[0][0], calls[-1][0])
    h = predictor.config.dt / predictor.config.substeps
    expected_impulse = h * sum(external.sum(1) for _, external in calls)
    expected_torque = h * sum(torque(position, external) for position, external in calls)
    torch.testing.assert_close(momentum(result) - momentum(state), expected_impulse,
                               atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(angular_momentum(result) - angular_momentum(state),
                               expected_torque, atol=3e-12, rtol=3e-12)


def test_coarsener_partition_actual_mass_and_inactive_group_gradients():
    state, _, mask = scene()
    state = state[1:].clone()
    mask = mask[1:]
    state[..., :2] = 0  # Three live coincident nodes create an unoccupied anchor slot.
    state[..., 5] *= 1e-10  # No physical or virtual mass floor is allowed.
    state.requires_grad_()
    features = torch.arange(8 * 16, dtype=state.dtype).reshape(1, 8, 16) / 100
    features.requires_grad_()
    coarse = ClosureCoarsener(4, 2)(features, state, mask)
    torch.testing.assert_close(coarse.weights.sum(-1), mask.to(state.dtype))
    torch.testing.assert_close(coarse.mass.sum(-1), state[..., 5].sum(-1), atol=0, rtol=2e-15)
    assert int(coarse.mask.sum()) == 2
    assert torch.equal(coarse.mask, coarse.mass > 0)
    assert torch.count_nonzero(coarse.shares[:, :, ~coarse.mask[0]]) == 0
    torch.testing.assert_close(coarse.shares.sum(1), coarse.mask.to(state.dtype))
    coarse_force = torch.tensor([[[0.3, -0.2], [0.4, 0.1], [-0.1, 0.6], [0.2, 0.3]]],
                                dtype=state.dtype).masked_fill(~coarse.mask.unsqueeze(-1), 0)
    prolonged = torch.bmm(coarse.shares, coarse_force)
    torch.testing.assert_close(prolonged.sum(1), coarse_force.sum(1), atol=1e-15, rtol=1e-15)
    loss = (coarse.features.square().sum() + coarse.position.sum()
            + coarse.velocity.square().sum() + coarse.shares.square().sum())
    loss.backward()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad[~mask]) == 0


@pytest.mark.parametrize("live_count", [1, 2, 3, 5])
def test_hierarchy_degenerate_and_padded_input_gradients_are_finite(live_count):
    state = torch.zeros(1, 8, 6, dtype=torch.float64)
    mask = torch.arange(8)[None] < live_count
    state[..., 4] = torch.where(mask, 0.1, 0)
    state[..., 5] = torch.where(mask, torch.arange(1, 9, dtype=state.dtype)[None] * 0.07, 0)
    state[..., 2:4] = torch.arange(16, dtype=state.dtype).reshape(1, 8, 2) / 100
    state = state.masked_fill(~mask.unsqueeze(-1), 0).requires_grad_()
    action = torch.full((1, 8, 2), 0.1, dtype=state.dtype).requires_grad_()
    predictor = model("hierarchy")
    force = predictor.internal_force(state, action, mask)
    probe = torch.arange(16, dtype=state.dtype).reshape(1, 8, 2) / 10
    (force * probe).sum().backward()
    assert torch.isfinite(force).all()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(action.grad).all()
    for parameter in predictor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(force.sum(1), torch.zeros(1, 2, dtype=state.dtype),
                               atol=1e-12, rtol=0)


def test_hierarchy_generic_input_derivatives_match_finite_differences():
    state, action, mask = scene()
    state = state[:1].clone().requires_grad_()
    action = action[:1].clone().requires_grad_()
    predictor = model("hierarchy")
    assert torch.autograd.gradcheck(
        lambda current, force: predictor.internal_force(current, force, mask[:1]),
        (state, action), eps=1e-6, atol=2e-5, rtol=2e-4, fast_mode=True,
    )


@pytest.mark.parametrize("coincident", [False, True])
def test_wrench_projection_finite_conservative_and_differentiable(coincident):
    state, action, mask = scene()
    if coincident:
        state[..., :2] = 0
    state.requires_grad_()
    force = (action * 13 + 0.37).requires_grad_()
    projected = project_internal_wrench(force, state, mask)
    torch.testing.assert_close(projected.sum(1), torch.zeros(2, 2, dtype=state.dtype),
                               atol=2e-15, rtol=0)
    torch.testing.assert_close(torque(state[..., :2], projected), torch.zeros(2, dtype=state.dtype),
                               atol=2e-15, rtol=0)
    projected.square().sum().backward()
    assert torch.isfinite(state.grad).all() and torch.isfinite(force.grad).all()
    assert torch.count_nonzero(projected[~mask]) == 0
    assert torch.count_nonzero(force.grad[~mask]) == 0


@pytest.mark.parametrize("kind", KINDS)
def test_single_disc_matches_simulator_in_free_drag_and_wall_contact(kind):
    state = torch.tensor(
        [[[0.13, -0.21, 0.27, -0.18, 0.08, 0.7]],
         [[0.945, -0.94, 0.3, -0.22, 0.08, 1.4]]], dtype=torch.float64,
    )
    action = torch.tensor([[[0.17, -0.09]], [[0.23, 0.14]]], dtype=state.dtype)
    mask = torch.ones(2, 1, dtype=torch.bool)
    expected = simulate(state, action, mask, Physics(global_coupling=1.0))
    actual = model(kind)(state, action, mask)
    torch.testing.assert_close(actual, expected, atol=3e-15, rtol=3e-15)


@pytest.mark.parametrize("substeps", [1, 4])
def test_constant_force_closed_form_and_small_physical_mass(substeps):
    state = torch.tensor([[[0.0, 0.0, 0.03, -0.04, 0.1, 0.01]]], dtype=torch.float64)
    action = torch.tensor([[[0.04, -0.02]]], dtype=state.dtype)
    mask = torch.ones(1, 1, dtype=torch.bool)
    predictor = model("prior", closure=False, substeps=substeps)
    actual = predictor(state, action, mask)
    acceleration = action / state[..., 5:6]
    expected_velocity = state[..., 2:4] + 0.05 * acceleration
    coefficient = (substeps + 1) / (2 * substeps)
    expected_position = state[..., :2] + 0.05 * state[..., 2:4] + coefficient * 0.05**2 * acceleration
    torch.testing.assert_close(actual[..., :2], expected_position, atol=1e-17, rtol=1e-15)
    torch.testing.assert_close(actual[..., 2:4], expected_velocity, atol=1e-17, rtol=1e-15)
    assert torch.equal(actual[..., 4:6], state[..., 4:6])


def test_single_disc_drag_geometric_sum():
    state = torch.tensor([[[0.13, -0.21, 0.27, -0.18, 0.08, 0.7]]], dtype=torch.float64)
    action = torch.zeros(1, 1, 2, dtype=state.dtype)
    mask = torch.ones(1, 1, dtype=torch.bool)
    predictor = model("prior")
    actual = predictor(state, action, mask)
    h = predictor.config.dt / predictor.config.substeps
    ratio = 1 - h * predictor.config.drag / state[..., 5:6]
    expected_velocity = ratio**4 * state[..., 2:4]
    expected_position = state[..., :2] + h * sum(ratio**step for step in range(1, 5)) * state[..., 2:4]
    torch.testing.assert_close(actual[..., :2], expected_position, atol=1e-16, rtol=1e-15)
    torch.testing.assert_close(actual[..., 2:4], expected_velocity, atol=1e-16, rtol=1e-15)


def test_law_contact_and_coupling_match_full_simulator():
    state, action, mask = scene()
    predictor = model("law")
    with torch.no_grad():
        predictor.rho.copy_(torch.log(torch.expm1(torch.tensor(1.0, dtype=state.dtype))))
    torch.testing.assert_close(predictor.learned_coupling(), torch.tensor(1.0, dtype=state.dtype))
    for _ in range(4):
        expected = simulate(state, action, mask, Physics(global_coupling=1.0))
        actual = predictor(state, action, mask)
        torch.testing.assert_close(actual, expected, atol=5e-15, rtol=5e-15)
        state = expected


def test_contact_is_repulsive_and_normal_clamp_matches_simulator():
    state = torch.tensor([[[0.0, 0.0, 0.2, 0.1, 0.1, 0.7],
                           [0.16, 0.0, -0.1, -0.2, 0.1, 1.4]]], dtype=torch.float64)
    action = torch.zeros(1, 2, 2, dtype=state.dtype)
    mask = torch.ones(1, 2, dtype=torch.bool)
    predictor = model("law")
    coupling = predictor.learned_coupling()
    expected = torch.tensor([[[-4.45, 0.0], [4.45, 0.0]]], dtype=state.dtype)
    expected -= coupling * (state[..., :2] - state[..., :2].mean(1, keepdim=True))
    torch.testing.assert_close(predictor.internal_force(state, action, mask), expected,
                               atol=2e-15, rtol=2e-15)
    state[0, 1, 0] = 1e-8
    with torch.no_grad():
        predictor.rho.copy_(torch.log(torch.expm1(torch.tensor(1.0, dtype=state.dtype))))
    torch.testing.assert_close(predictor(state, action, mask),
                               simulate(state, action, mask, Physics(global_coupling=1.0)),
                               atol=1e-14, rtol=1e-14)


def test_fixed_walls_break_arbitrary_translation_and_do_not_attract():
    state = torch.tensor([[[0.8, 0.0, 0.2, 0.0, 0.1, 0.7]]], dtype=torch.float64)
    action = torch.zeros(1, 1, 2, dtype=state.dtype)
    mask = torch.ones(1, 1, dtype=torch.bool)
    shifted = state.clone()
    shifted[..., 0] += 0.15
    predictor = model("prior")
    assert not torch.allclose(predictor(state, action, mask)[..., 2:4],
                              predictor(shifted, action, mask)[..., 2:4])
    shifted[..., 0] = 0.93
    shifted[..., 2] = -3
    torch.testing.assert_close(predictor.external_force(shifted, action, mask),
                               -predictor.config.drag * shifted[..., 2:4])


def test_local_cutoff_is_strict_and_global_head_reaches_distant_pairs():
    state = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.1, 0.7],
                           [0.6, 0.0, 0.0, 0.0, 0.1, 1.4]]], dtype=torch.float64)
    action = torch.zeros(1, 2, 2, dtype=state.dtype)
    mask = torch.ones(1, 2, dtype=torch.bool)
    local = model("local")
    global_model = model("global")
    assert torch.count_nonzero(local.internal_force(state, action, mask)) == 0
    assert torch.count_nonzero(global_model.internal_force(state, action, mask)) > 0
    state[..., 1, 0] = 0.599
    assert torch.count_nonzero(local.internal_force(state, action, mask)) > 0


def checkpoint(predictor: ClosureModel):
    return {"format": "ripii-closure-v1", "config": asdict(predictor.config),
            "model_state": predictor.state_dict(), "selection": "fixture",
            "completed_updates": 0, "seed": 101, "source_sha256": "a" * 64,
            "protocol_sha256": "b" * 64}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_roundtrip_preserves_predictions_dtype_metadata_and_rng(kind, dtype, tmp_path):
    predictor = model(kind).to(dtype=dtype)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint(predictor), path)
    rng = torch.get_rng_state().clone()
    restored, payload = load_closure_checkpoint(path)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not restored.training
    assert restored.config == predictor.config
    assert payload["completed_updates"] == 0 and payload["selection"] == "fixture"
    state, action, mask = scene(dtype)
    assert torch.equal(restored(state, action, mask), predictor(state, action, mask))
    for parameter in restored.parameters():
        assert parameter.dtype == dtype


@pytest.mark.parametrize(
    "corruption",
    ["legacy_format", "missing_header", "extra_config", "missing_config", "negative_config",
     "kind_relabel", "closure_relabel", "substeps_relabel", "non_dict_state", "missing_weight",
     "extra_weight", "shape", "nan", "infinite", "integer", "half", "mixed_dtype", "signature"],
)
def test_checkpoint_rejects_schema_semantic_and_weight_corruption(corruption, tmp_path):
    payload = copy.deepcopy(checkpoint(model("global")))
    weights = payload["model_state"]
    name = "encoder.0.weight"
    if corruption == "legacy_format":
        payload["format"] = "ripii-world-checkpoint-v1"
    elif corruption == "missing_header":
        del payload["config"]
    elif corruption == "extra_config":
        payload["config"]["router"] = True
    elif corruption == "missing_config":
        del payload["config"]["dt"]
    elif corruption == "negative_config":
        payload["config"]["drag"] = -1
    elif corruption == "kind_relabel":
        payload["config"]["kind"] = "hierarchy"  # All parameter keys/shapes still match.
    elif corruption == "closure_relabel":
        payload["config"]["closure"] = False
    elif corruption == "substeps_relabel":
        payload["config"]["substeps"] = 1
    elif corruption == "non_dict_state":
        payload["model_state"] = []
    elif corruption == "missing_weight":
        del weights[name]
    elif corruption == "extra_weight":
        weights["unknown"] = torch.tensor(1.0)
    elif corruption == "shape":
        weights[name] = weights[name][:1]
    elif corruption == "nan":
        weights[name][0, 0] = math.nan
    elif corruption == "infinite":
        weights[name][0, 0] = math.inf
    elif corruption == "integer":
        weights[name] = weights[name].long()
    elif corruption == "half":
        weights[name] = weights[name].half()
    elif corruption == "mixed_dtype":
        weights[name] = weights[name].float()
    elif corruption == "signature":
        weights["_config_signature"][0] ^= 1
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises((TypeError, ValueError)):
        load_closure_checkpoint(path)


def test_checkpoint_rejects_nonregular_or_unreadable_files(tmp_path):
    missing = tmp_path / "missing.pt"
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    symlink = tmp_path / "link.pt"
    symlink.symlink_to(bad)
    for path in (missing, bad, symlink, Path(tmp_path)):
        with pytest.raises(ValueError):
            load_closure_checkpoint(path)


def test_nonfinite_model_output_fails_explicitly():
    state, action, mask = scene()
    predictor = model("local")
    with torch.no_grad():
        predictor.encoder[0].weight.fill_(math.nan)
    with pytest.raises(FloatingPointError):
        predictor(state, action, mask)
