from __future__ import annotations

import json

import pytest
import torch

from ripii.world.mechanics_audit import (
    conservation_bounds,
    diagnostic_model,
    fixture_specs,
    fixture_tensors,
    run_audit,
    verify_audit,
)
from ripii.world.models import rollout
from ripii.world.physics import make_dataset


def test_cauchy_bound_is_attained_by_mass_proportional_coordinate_errors():
    # The errors [0.1, 0.2] are proportional to masses [0.7, 1.4].
    # Constant-velocity predictions therefore attain the Cauchy bound exactly.
    initial = torch.tensor(
        [[[0, 0, 0, 0, 0.07, 0.7], [0.5, 0, 0, 0, 0.07, 1.4]]],
        dtype=torch.float64,
    )
    target = initial.clone()
    target[0, :, 0] += torch.tensor([0.1, 0.2], dtype=torch.float64)
    target[0, :, 2] += torch.tensor([0.2, 0.4], dtype=torch.float64)
    states = torch.stack([initial, target], 1)
    actions = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    bounds = conservation_bounds(
        states, actions, torch.ones(1, 2, dtype=torch.bool), 0.05
    )
    actual_position_rmse = ((initial[..., :2] - target[..., :2]).square().mean()).sqrt()
    actual_velocity_rmse = (
        (initial[..., 2:4] - target[..., 2:4]).square().mean()
    ).sqrt()
    torch.testing.assert_close(
        bounds["position_rmse_lower_bound"], actual_position_rmse, atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        bounds["velocity_rmse_lower_bound"], actual_velocity_rmse, atol=1e-14, rtol=0
    )


@pytest.mark.parametrize(
    "defect", ["mass_drift", "tiny_mass", "numeric_mask", "nonfinite", "padding", "dt"]
)
def test_bounds_reject_inputs_that_invalidate_the_derivation(defect):
    state, actions, mask = fixture_tensors(fixture_specs()[0])
    states = state[:, None].repeat(1, actions.shape[1] + 1, 1, 1)
    dt = 0.05
    if defect == "mass_drift":
        states[0, 1, 0, 5] += 0.1
    elif defect == "tiny_mass":
        states[0, :, 0, 5] = 1e-9
    elif defect == "numeric_mask":
        mask = mask.float()
    elif defect == "nonfinite":
        actions[0, 0, 0, 0] = float("nan")
    elif defect == "padding":
        states[0, :, 7, 0] = 1
    elif defect == "dt":
        dt = True
    with pytest.raises((ValueError, FloatingPointError)):
        conservation_bounds(states, actions, mask, dt)


@pytest.mark.parametrize("variant", ["ripii_mr", "ripii_mr_v2"])
def test_zero_padded_multistep_input_and_mechanism_gradients_are_finite(variant):
    data = make_dataset("train", 4, 2, 1913)
    state = data["states"][:, 0].double().requires_grad_()
    actions = data["actions"].double().requires_grad_()
    mask = data["mask"]
    model = diagnostic_model(variant).train()
    prediction = rollout(model, state, actions, mask)
    prediction[:, 1:, :, :4].square().mean().backward()
    assert state.grad is not None and actions.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(actions.grad).all()
    assert (state.grad[~mask] == 0).all()
    assert (actions.grad.masked_select(~mask[:, None, :, None]) == 0).all()
    for module in (
        model.multiresolution.fine_impulse,
        model.multiresolution.coarse_impulse,
        model.multiresolution.router,
    ):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(
            torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_diagnostic_preserves_source_weights_results_and_detects_tampering(tmp_path):
    output = tmp_path / "audit"
    previous_rng = torch.random.get_rng_state().clone()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    receipt = run_audit(output)
    assert torch.equal(torch.random.get_rng_state(), previous_rng)
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_deterministic
    assert receipt["fixtures"] == 9
    assert receipt["status"] == "verified_integrity_and_completion"
    result = json.loads((output / "measurements.json").read_text())
    assert result["training_updates"] == 0
    by_name = {row["name"]: row for row in result["fixtures"]}
    # Removing drag or matching the force integrator removes these exact
    # single-particle obstructions. This is not a trained model comparison.
    assert by_name["drag_light"]["bounds"]["position_rmse_lower_bound"] > 0
    assert by_name["free_motion"]["bounds"]["position_rmse_lower_bound"] < 1e-12
    assert by_name["force_four_substeps"]["bounds"]["position_rmse_lower_bound"] > 0
    assert by_name["force_one_substep"]["bounds"]["position_rmse_lower_bound"] < 1e-12
    routes = {(row["variant"], row["route_on"]): row for row in result["routing_probe"]}
    assert routes[("ripii_mr", False)]["coarsener_calls"] == 1
    assert routes[("ripii_mr_v2", False)]["coarsener_calls"] == 0
    assert routes[("ripii_mr_v2", False)]["router_task_gradient_l1"] == 0
    assert routes[("ripii_mr_v2", True)]["router_task_gradient_l1"] > 0
    with pytest.raises(FileExistsError):
        run_audit(output)
    report = output / "report.md"
    original = report.read_bytes()
    report.write_text("changed result")
    with pytest.raises(ValueError, match="integrity"):
        verify_audit(output)
    report.write_bytes(original)
    (output / "unexpected.txt").write_text("unmanifested")
    with pytest.raises(ValueError, match="file set"):
        verify_audit(output)
