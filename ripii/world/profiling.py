from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass

import torch

from ..utils.statistics import bootstrap_mean_ci, descriptive_summary
from .models import rollout


@dataclass(frozen=True)
class ProfileConfig:
    warmup: int = 5
    repeats: int = 30
    profile_flops: bool = True

    def validate(self) -> None:
        if (
            not isinstance(self.warmup, int)
            or isinstance(self.warmup, bool)
            or self.warmup < 1
            or not isinstance(self.repeats, int)
            or isinstance(self.repeats, bool)
            or self.repeats < 5
            or not isinstance(self.profile_flops, bool)
        ):
            raise ValueError("profiling requires >=1 warmup and >=5 measured repeats")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _recognized_flops(model, state, actions, mask) -> int:
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as trace:
        rollout(model, state, actions, mask)
    return int(sum(event.flops or 0 for event in trace.key_averages()))


@torch.no_grad()
def profile_rollout(
    model,
    data: dict[str, torch.Tensor],
    config: ProfileConfig = ProfileConfig(),
) -> dict:
    config.validate()
    required = {"states", "actions", "mask"}
    if not required <= data.keys():
        raise ValueError(f"profiling data is missing {sorted(required - data.keys())}")
    state, actions, mask = data["states"][:, 0], data["actions"], data["mask"]
    if (
        state.ndim != 3
        or actions.ndim != 4
        or actions.shape[0] != state.shape[0]
        or actions.shape[2] != state.shape[1]
        or mask.shape != state.shape[:2]
        or mask.dtype != torch.bool
        or not torch.isfinite(state).all()
        or not torch.isfinite(actions).all()
    ):
        raise ValueError("invalid profiling tensors")
    model.eval()
    device = state.device
    for _ in range(config.warmup):
        rollout(model, state, actions, mask)
    _synchronize(device)
    timings = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(config.repeats):
        start = time.perf_counter()
        result = rollout(model, state, actions, mask)
        _synchronize(device)
        timings.append(time.perf_counter() - start)
    if not torch.isfinite(result).all() or not all(
        math.isfinite(value) and value > 0 for value in timings
    ):
        raise FloatingPointError("profiling produced non-finite output or timing")
    interval = bootstrap_mean_ci(timings, seed=83_011)
    summary = descriptive_summary(timings)
    scene_steps = state.shape[0] * actions.shape[1]
    recognized_flops = (
        _recognized_flops(model, state, actions, mask) if config.profile_flops else None
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "format": "ripii-world-profile-v1",
        "warmup_runs": config.warmup,
        "measured_runs": config.repeats,
        "batch_scenes": state.shape[0],
        "rollout_steps": actions.shape[1],
        "timing_seconds": summary,
        "bootstrap_95_ci_mean_seconds": [interval[0], interval[1]],
        "scene_steps_per_second_at_mean": scene_steps / summary["mean"],
        "recognized_operator_flops_per_rollout": recognized_flops,
        "flop_boundary": (
            "PyTorch profiler count for recognized operators; a lower-bound proxy, "
            "not a complete hardware-independent FLOP accounting"
        ),
        "peak_accelerator_memory_bytes": peak_memory,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
        },
    }
