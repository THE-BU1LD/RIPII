"""Public inference admission, without modifying the frozen model or studies."""

import numpy as np
import pytest
import torch

from ripii.world.inference import WorldPredictor
from ripii.world.models import WorldModel
from ripii.world.physics import sample_scene


@pytest.fixture
def predictor():
    return WorldPredictor(WorldModel("graph", hidden=16, max_objects=8), {})


def inputs(steps=None):
    state, mask = sample_scene(torch.Generator().manual_seed(83), 3)
    action_shape = (1, 8, 2) if steps is None else (1, steps, 8, 2)
    return state[None], torch.zeros(action_shape), mask[None]


def forbid_forward(predictor, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid input reached the model")

    monkeypatch.setattr(predictor.model, "forward", fail)
    return calls


@pytest.mark.parametrize("sequence", [False, True], ids=["predict", "rollout"])
@pytest.mark.parametrize("mask_kind", ["integer", "fractional", "nan"])
def test_nonboolean_masks_are_rejected_before_coercion(predictor, monkeypatch, sequence, mask_kind):
    state, action, mask = inputs(2 if sequence else None)
    mask = mask.float()
    if mask_kind == "integer":
        mask = mask.long()
    elif mask_kind == "fractional":
        mask[0, 0] = 0.5
    else:
        mask[0, 0] = float("nan")
    calls = forbid_forward(predictor, monkeypatch)
    infer = predictor.rollout if sequence else predictor.predict
    with pytest.raises(TypeError, match="boolean"):
        infer(state, action, mask)
    assert not calls


@pytest.mark.parametrize("sequence", [False, True], ids=["predict", "rollout"])
@pytest.mark.parametrize(
    "kind,error,message",
    [
        ("integer_state", TypeError, "floating-point"),
        ("integer_action", TypeError, "floating-point"),
        ("complex_state", TypeError, "floating-point"),
        ("state_shape", ValueError, "nonempty shape"),
        ("mask_shape", ValueError, "mask shape"),
        ("action_shape", ValueError, "action"),
        ("nonfinite_state", FloatingPointError, "non-finite"),
        ("float32_overflow", FloatingPointError, "non-finite"),
        ("empty_scene", ValueError, "live object"),
        ("zero_mass", ValueError, "positive radius and mass"),
        ("padded_state", ValueError, "padded state"),
        ("padded_action", ValueError, "padded action"),
    ],
)
def test_invalid_inputs_fail_before_any_model_call(predictor, monkeypatch, sequence, kind, error, message):
    state, action, mask = inputs(2 if sequence else None)
    if kind == "integer_state":
        state = state.long()
    elif kind == "integer_action":
        action = action.long()
    elif kind == "complex_state":
        state = state.to(torch.complex64) + 1j
    elif kind == "state_shape":
        state = state[..., :5]
    elif kind == "mask_shape":
        mask = mask[:, :7]
    elif kind == "action_shape":
        action = action[..., :7, :]
    elif kind == "nonfinite_state":
        state[0, 0, 0] = float("nan")
    elif kind == "float32_overflow":
        state = state.double()
        state[0, 0, 0] = 1e300
    elif kind == "empty_scene":
        mask[:] = False
    elif kind == "zero_mass":
        state[0, 0, 5] = 0
    elif kind == "padded_state":
        state[0, -1, 0] = 1
    elif kind == "padded_action":
        action[..., -1, 0] = 1
    calls = forbid_forward(predictor, monkeypatch)
    infer = predictor.rollout if sequence else predictor.predict
    with pytest.raises(error, match=message):
        infer(state, action, mask)
    assert not calls


def test_late_nonfinite_action_rejects_whole_rollout_before_first_step(predictor, monkeypatch):
    state, actions, mask = inputs(3)
    actions[:, -1, 0, 0] = float("inf")
    calls = forbid_forward(predictor, monkeypatch)
    with pytest.raises(FloatingPointError, match="non-finite"):
        predictor.rollout(state, actions, mask)
    assert not calls


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_objects", "invalid_mass", "empty_batch", "numeric_mask"])
def test_zero_step_rollout_still_validates_complete_input(predictor, monkeypatch, kind):
    state, actions, mask = inputs(0)
    if kind == "nonfinite":
        state[0, 0, 0] = float("nan")
    elif kind == "wrong_objects":
        actions = actions[:, :, :7]
    elif kind == "invalid_mass":
        state[0, 0, 5] = -1
    elif kind == "empty_batch":
        state, actions, mask = state[:0], actions[:0], mask[:0]
    else:
        mask = mask.float()
    calls = forbid_forward(predictor, monkeypatch)
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        predictor.rollout(state, actions, mask)
    assert not calls


def test_valid_zero_step_rollout_returns_initial_state_without_model_call(predictor, monkeypatch):
    state, actions, mask = inputs(0)
    calls = forbid_forward(predictor, monkeypatch)
    result = predictor.rollout(state, actions, mask)
    torch.testing.assert_close(result, state[:, None], rtol=0, atol=0)
    assert result.shape == (1, 1, 8, 6)
    assert not result.requires_grad
    assert not calls


@pytest.mark.parametrize("sequence", [False, True], ids=["predict", "rollout"])
def test_valid_numpy_float64_inputs_keep_existing_float32_predictions(predictor, sequence):
    state, action, mask = inputs(2 if sequence else None)
    infer = predictor.rollout if sequence else predictor.predict
    expected = infer(state, action, mask)
    actual = infer(
        state.numpy().astype(np.float64),
        action.numpy().astype(np.float64),
        mask.numpy(),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert not actual.requires_grad
