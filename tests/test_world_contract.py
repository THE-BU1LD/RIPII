from __future__ import annotations

import numpy as np
import pytest
import torch

from ripii.world.inference import WorldPredictor
from ripii.world.models import MODEL_VARIANTS, WorldModel, rollout
from ripii.world.physics import sample_scene


def inputs(horizon: int = 0):
    state, mask = sample_scene(torch.Generator().manual_seed(1901), 3)
    return state[None], torch.zeros(1, horizon, 8, 2), mask[None]


@pytest.mark.parametrize("variant", MODEL_VARIANTS)
def test_zero_step_rollout_validates_and_returns_initial_state_without_execution(
    variant,
):
    model = WorldModel(variant, hidden=16)
    state, actions, mask = inputs()
    calls = []
    model.register_forward_hook(lambda *_args: calls.append(True))
    actual = rollout(model, state, actions, mask)
    assert actual.shape == (1, 1, 8, 6)
    assert torch.equal(actual[:, 0], state)
    assert calls == []


@pytest.mark.parametrize(
    ("defect", "error"),
    [
        ("state_shape", ValueError),
        ("action_objects", ValueError),
        ("action_features", ValueError),
        ("mask_shape", ValueError),
        ("mask_type", TypeError),
        ("state_type", TypeError),
        ("action_type", TypeError),
        ("mixed_dtype", TypeError),
        ("empty_scene", ValueError),
        ("empty_batch", ValueError),
        ("nonfinite", FloatingPointError),
        ("radius", ValueError),
        ("mass", ValueError),
        ("padding", ValueError),
    ],
)
def test_zero_step_rollout_rejects_invalid_requests(defect, error):
    model = WorldModel("graph", hidden=16)
    state, actions, mask = inputs()
    if defect == "state_shape":
        state = state[..., :5]
    elif defect == "action_objects":
        actions = torch.empty(1, 0, 7, 2)
    elif defect == "action_features":
        actions = torch.empty(1, 0, 8, 3)
    elif defect == "mask_shape":
        mask = mask[:, :7]
    elif defect == "mask_type":
        mask = mask.float()
    elif defect == "state_type":
        state = state.long()
    elif defect == "action_type":
        actions = actions.long()
    elif defect == "mixed_dtype":
        actions = actions.double()
    elif defect == "empty_scene":
        mask[:] = False
    elif defect == "empty_batch":
        state, actions, mask = state[:0], actions[:0], mask[:0]
    elif defect == "nonfinite":
        state[0, 0, 0] = float("nan")
    elif defect == "radius":
        state[0, 0, 4] = 0
    elif defect == "mass":
        state[0, 0, 5] = -1
    elif defect == "padding":
        state[~mask] = 1e30
    with pytest.raises(error):
        rollout(model, state, actions, mask)


@pytest.mark.parametrize("defect", ["nonfinite", "padding"])
def test_rollout_checks_all_actions_before_first_forward(defect):
    model = WorldModel("ripii_mr_v2", hidden=16)
    state, actions, mask = inputs(horizon=2)
    calls = []
    model.register_forward_hook(lambda *_args: calls.append(True))
    if defect == "nonfinite":
        actions[0, 1, 0, 0] = float("inf")
        error = FloatingPointError
    else:
        actions[0, 1, 7, 0] = 1
        error = ValueError
    with pytest.raises(error):
        rollout(model, state, actions, mask)
    assert calls == []
    assert model.multiresolution.compute_counters()["coarse_compute_calls"] == 0


@pytest.mark.parametrize("variant", MODEL_VARIANTS)
@pytest.mark.parametrize("field", ["state", "action"])
def test_forward_enforces_documented_zero_padding(variant, field):
    model = WorldModel(variant, hidden=16)
    state, actions, mask = inputs(horizon=1)
    action = actions[:, 0]
    if field == "state":
        state[~mask] = 1e30
    else:
        action[~mask] = 1e30
    with pytest.raises(ValueError, match="padding"):
        model(state, action, mask)


@pytest.mark.parametrize("method", ["predict", "rollout"])
@pytest.mark.parametrize("kind", ["integer", "float", "nan", "two"])
def test_predictor_rejects_nonboolean_masks_before_conversion(method, kind):
    predictor = WorldPredictor(WorldModel("graph", hidden=16), {})
    state, actions, mask = inputs(horizon=1)
    bad_mask = mask.numpy().astype(np.int64 if kind == "integer" else np.float32)
    if kind == "nan":
        bad_mask[0, 0] = np.nan
    elif kind == "two":
        bad_mask[0, 0] = 2
    action = actions[:, 0] if method == "predict" else actions
    with pytest.raises(TypeError, match="boolean"):
        getattr(predictor, method)(state.numpy(), action.numpy(), bad_mask)


@pytest.mark.parametrize("field", ["state", "action"])
def test_predictor_rejects_nonfloating_state_or_actions(field):
    predictor = WorldPredictor(WorldModel("graph", hidden=16), {})
    state, actions, mask = inputs(horizon=1)
    action = actions[:, 0]
    if field == "state":
        state = state.long()
    else:
        action = action.long()
    with pytest.raises(TypeError, match="floating"):
        predictor.predict(state, action, mask)


def test_predictor_accepts_boolean_numpy_masks_and_converts_floating_precision():
    predictor = WorldPredictor(WorldModel("graph", hidden=16), {})
    state, actions, mask = inputs(horizon=2)
    actual = predictor.rollout(
        state.double().numpy(), actions.double().numpy(), mask.numpy()
    )
    assert actual.shape == (1, 3, 8, 6)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[:, :, ~mask[0]], torch.zeros_like(actual[:, :, ~mask[0]]))


def test_model_accepts_matching_double_tensors():
    model = WorldModel("ripii_mr", hidden=16).double()
    state, actions, mask = inputs(horizon=2)
    actual = rollout(model, state.double(), actions.double(), mask)
    assert actual.dtype == torch.float64
    assert torch.isfinite(actual).all()
