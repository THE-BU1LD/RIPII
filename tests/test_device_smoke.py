import json

import pytest
import torch

from scripts import check_device_smoke


def _alter_forward(monkeypatch, transform) -> None:
    original = check_device_smoke.WorldModel.forward
    calls = 0

    def forward(self, state, action, mask):
        nonlocal calls
        calls += 1
        return transform(original(self, state, action, mask), calls)

    monkeypatch.setattr(check_device_smoke.WorldModel, "forward", forward)


def test_cpu_device_smoke() -> None:
    result = check_device_smoke.check("cpu")
    assert result["status"] == "PASS"
    assert result["maximum_repeat_error"] <= 1e-6
    assert result["finite_gradient_tensors"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_one_nonfinite_gradient_among_finite_gradients(monkeypatch, value):
    original = check_device_smoke.WorldModel

    def poisoned_model(*args, **kwargs):
        model = original(*args, **kwargs)
        next(model.parameters()).register_hook(
            lambda gradient: torch.full_like(gradient, value)
        )
        return model

    monkeypatch.setattr(check_device_smoke, "WorldModel", poisoned_model)
    with pytest.raises(RuntimeError, match="finite/repeatable gradient contract"):
        check_device_smoke.check("cpu")


def test_rejects_nonfinite_first_output_even_when_second_recovers(monkeypatch):
    _alter_forward(
        monkeypatch,
        lambda output, call: output * float("nan") if call == 1 else output,
    )
    with pytest.raises(RuntimeError, match="finite/repeatable gradient contract"):
        check_device_smoke.check("cpu")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_second_output(monkeypatch, value):
    _alter_forward(
        monkeypatch, lambda output, call: output * value if call == 2 else output
    )
    with pytest.raises(RuntimeError, match="finite/repeatable gradient contract"):
        check_device_smoke.check("cpu")


def test_rejects_overflowing_loss_with_finite_output_and_gradients(monkeypatch):
    _alter_forward(monkeypatch, lambda output, _call: output * 0 + 1e20)
    with pytest.raises(RuntimeError, match="finite/repeatable gradient contract"):
        check_device_smoke.check("cpu")


def test_rejects_finite_nonrepeatable_outputs(monkeypatch):
    _alter_forward(
        monkeypatch, lambda output, call: output + (0.1 if call == 2 else 0)
    )
    with pytest.raises(RuntimeError, match="finite/repeatable gradient contract"):
        check_device_smoke.check("cpu")


def test_unused_parameters_do_not_require_a_gradient(monkeypatch):
    original = check_device_smoke.WorldModel

    def model_with_unused_parameter(*args, **kwargs):
        model = original(*args, **kwargs)
        model.register_parameter("unused_probe", torch.nn.Parameter(torch.zeros(1)))
        return model

    monkeypatch.setattr(check_device_smoke, "WorldModel", model_with_unused_parameter)
    assert check_device_smoke.check("cpu")["status"] == "PASS"
