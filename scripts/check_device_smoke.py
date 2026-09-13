"""Exercise finite forward/backward and repeatability on an explicit device."""

from __future__ import annotations

import argparse
import json

import torch

from ripii.world.models import WorldModel


def check(device_name: str) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was required but is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(9127)
    model = WorldModel("graph", hidden=16, max_objects=5).to(device)
    state = torch.zeros(2, 5, 6, device=device)
    state[..., 4:] = 1.0
    action = torch.zeros(2, 5, 2, device=device)
    mask = torch.ones(2, 5, dtype=torch.bool, device=device)
    first = model(state, action, mask)
    second = model(state, action, mask)
    loss = first.square().mean()
    loss.backward()
    maximum_repeat_error = float((first - second).abs().max().detach().cpu())
    gradient_tensors = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if (
        not torch.isfinite(first).all()
        or maximum_repeat_error > 1e-6
        or gradient_tensors == 0
    ):
        raise RuntimeError("device smoke violated finite/repeatable gradient contract")
    return {
        "status": "PASS",
        "device": device_name,
        "maximum_repeat_error": maximum_repeat_error,
        "finite_gradient_tensors": gradient_tensors,
        "torch": str(torch.__version__),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    print(json.dumps(check(parser.parse_args().device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
