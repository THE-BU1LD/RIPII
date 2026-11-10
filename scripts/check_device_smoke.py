"""Exercise finite forward/backward and repeatability on an explicit device."""

from __future__ import annotations

import argparse
import json
import math

import torch

from ripii.world.models import WorldModel


def _release_device_cache(device_name: str) -> None:
    """Release allocator cache without changing device safety limits."""
    if device_name == "mps":
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device_name == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def check(device_name: str) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was required but is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(9127)

    # Use the constructor's smallest valid graph configuration. The contract is
    # device-path forward/backward repeatability, not stress-testing allocator
    # capacity on a shared hosted runner.
    _release_device_cache(device_name)
    model = WorldModel("graph", hidden=8, max_objects=5).to(device)
    state = torch.zeros(1, 5, 6, device=device)
    state[..., 4:] = 1.0
    action = torch.zeros(1, 5, 2, device=device)
    mask = torch.ones(1, 5, dtype=torch.bool, device=device)

    first = model(state, action, mask)
    first_snapshot = first.detach().cpu()
    del first
    _release_device_cache(device_name)

    second = model(state, action, mask)
    second_snapshot = second.detach().cpu()
    maximum_repeat_error = float((first_snapshot - second_snapshot).abs().max())

    loss = second.square().mean()
    if (
        not torch.isfinite(first_snapshot).all()
        or not torch.isfinite(second_snapshot).all()
        or not torch.isfinite(loss)
        or not math.isfinite(maximum_repeat_error)
        or maximum_repeat_error > 1e-6
    ):
        raise RuntimeError("device smoke violated finite/repeatable gradient contract")

    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("device smoke violated finite/repeatable gradient contract")
    return {
        "status": "PASS",
        "device": device_name,
        "maximum_repeat_error": maximum_repeat_error,
        "finite_gradient_tensors": len(gradients),
        "torch": str(torch.__version__),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    print(json.dumps(check(parser.parse_args().device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
