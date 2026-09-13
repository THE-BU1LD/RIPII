"""Fail CI on gross parameter-count or CPU-forward performance regressions."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch

from ripii.world.models import WorldModel


def check(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("format") != "ripii-ci-performance-contract-v1":
        raise ValueError("unknown CI performance contract")
    torch.set_num_threads(contract["torch_threads"])
    model = WorldModel(
        contract["variant"],
        hidden=contract["hidden"],
        max_objects=contract["max_objects"],
    ).eval()
    state = torch.zeros(
        contract["batch_scenes"], contract["max_objects"], 6, dtype=torch.float32
    )
    state[..., 4:] = 1.0
    action = torch.zeros(
        contract["batch_scenes"], contract["max_objects"], 2, dtype=torch.float32
    )
    mask = torch.ones(
        contract["batch_scenes"], contract["max_objects"], dtype=torch.bool
    )
    with torch.no_grad():
        for _ in range(contract["warmup"]):
            model(state, action, mask)
        elapsed = []
        for _ in range(contract["repeats"]):
            started = time.perf_counter()
            output = model(state, action, mask)
            elapsed.append(time.perf_counter() - started)
    parameters = sum(item.numel() for item in model.parameters() if item.requires_grad)
    mean_seconds = sum(elapsed) / len(elapsed)
    if not torch.isfinite(output).all() or not math.isfinite(mean_seconds):
        raise FloatingPointError("CI performance smoke produced non-finite output")
    failures = []
    if parameters > contract["max_trainable_parameters"]:
        failures.append("trainable parameter budget exceeded")
    if mean_seconds > contract["max_mean_forward_seconds"]:
        failures.append("mean CPU forward-time budget exceeded")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "parameters": parameters,
        "mean_forward_seconds": mean_seconds,
        "max_forward_seconds": max(elapsed),
        "contract": contract,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "machine": platform.machine(),
        },
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "contract",
        type=Path,
        nargs="?",
        default=Path("research/protocols/ci_performance_contract_v1.json"),
    )
    print(json.dumps(check(parser.parse_args().contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
