from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

STATE_DIM = 6  # x, y, vx, vy, radius, mass; the square arena is [-1, 1]^2.
SPLITS = ("train", "validation", "test", "more_objects", "composition", "fast")
OFFSETS = {name: i * 1_000_003 for i, name in enumerate(SPLITS)}


@dataclass(frozen=True)
class Physics:
    dt: float = 0.05
    substeps: int = 4
    stiffness: float = 100.0
    damping: float = 1.5
    drag: float = 0.06
    global_coupling: float = 0.0

    def __post_init__(self):
        numeric = (
            self.dt,
            self.stiffness,
            self.damping,
            self.drag,
            self.global_coupling,
        )
        if (
            not all(math.isfinite(value) for value in numeric)
            or self.dt <= 0
            or not isinstance(self.substeps, int)
            or self.substeps < 1
            or min(
                self.stiffness,
                self.damping,
                self.drag,
                self.global_coupling,
            )
            < 0
        ):
            raise ValueError("invalid physics parameters")

    def as_dict(self):
        return asdict(self)


def simulate(
    state: torch.Tensor,
    action: torch.Tensor,
    mask: torch.Tensor,
    physics: Physics = Physics(),
) -> torch.Tensor:
    """Semi-implicit integration of damped soft-disc contacts and wall contacts.

    Actions are forces, held constant across integration substeps. Padded objects
    exert no forces. Contact and optional all-pairs harmonic forces are symmetric,
    so pair momentum is conserved in the absence of external forces, walls, and drag.
    """
    if state.ndim != 3 or state.shape[-1] != STATE_DIM:
        raise ValueError("state must have shape [batch, objects, 6]")
    if action.shape != (*state.shape[:2], 2) or mask.shape != state.shape[:2]:
        raise ValueError("action and mask shapes must match the state")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if not state.is_floating_point() or not action.is_floating_point():
        raise TypeError("state and action must be floating-point tensors")
    if state.device != action.device or state.device != mask.device:
        raise ValueError("state, action, and mask must share a device")
    if not torch.isfinite(state).all() or not torch.isfinite(action).all():
        raise FloatingPointError("non-finite simulator input")
    if not mask.any(dim=1).all():
        raise ValueError("every scene must contain at least one live object")
    live_properties = state[..., 4:][mask]
    if (live_properties <= 0).any():
        raise ValueError("live objects require positive radius and mass")
    position, velocity = state[..., :2].clone(), state[..., 2:4].clone()
    radius, mass = state[..., 4:5], state[..., 5:6].clamp_min(0.1)
    live = mask.unsqueeze(-1).to(state.dtype)
    pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    pairs = pairs & ~torch.eye(state.shape[-2], device=state.device, dtype=torch.bool)
    h = physics.dt / physics.substeps
    for _ in range(physics.substeps):
        relative = position.unsqueeze(-2) - position.unsqueeze(-3)
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = relative / distance
        overlap = radius.unsqueeze(-2) + radius.unsqueeze(-3) - distance
        closing = ((velocity.unsqueeze(-2) - velocity.unsqueeze(-3)) * normal).sum(
            -1, keepdim=True
        )
        strength = (physics.stiffness * overlap - physics.damping * closing).clamp_min(
            0
        )
        contact = (overlap > 0) & pairs.unsqueeze(-1)
        force = (normal * strength * contact).sum(-2)
        if physics.global_coupling:
            # A momentum-preserving long-range regime. Dividing by the live-object
            # count keeps the acceleration scale comparable across scene sizes.
            count = mask.sum(-1, keepdim=True).clamp_min(1).unsqueeze(-1)
            global_force = (
                -physics.global_coupling
                * (relative * pairs.unsqueeze(-1)).sum(-2)
                / count
            )
            force = force + global_force
        low, high = -1 + radius, 1 - radius
        wall_low = (
            physics.stiffness * (low - position) - physics.damping * velocity
        ).clamp_min(0)
        wall_high = (
            physics.stiffness * (position - high) + physics.damping * velocity
        ).clamp_min(0)
        force = force + wall_low * (position < low) - wall_high * (position > high)
        force = force + action - physics.drag * velocity
        velocity = velocity + h * force / mass * live
        position = position + h * velocity * live
    result = torch.cat([position, velocity, radius, mass], -1) * live
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite simulator output")
    return result


def sample_scene(
    generator: torch.Generator,
    objects: int,
    max_objects: int = 8,
    composition: bool = False,
    speed: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(objects, int)
        or not isinstance(max_objects, int)
        or not 1 <= objects <= max_objects <= 16
        or not math.isfinite(speed)
        or speed < 0
    ):
        raise ValueError("require 1 <= objects <= max_objects <= 16")
    state = torch.zeros(max_objects, STATE_DIM)
    mask = torch.arange(max_objects) < objects
    # Training sees small/light, small/heavy, large/light. Large/heavy is held out.
    properties = [(0.07, 0.7), (0.07, 1.4), (0.11, 0.7)]
    for i in range(objects):
        radius, mass = (
            (0.11, 1.4)
            if composition
            else properties[int(torch.randint(3, (), generator=generator))]
        )
        for _ in range(1000):
            point = (torch.rand(2, generator=generator) * 2 - 1) * (0.9 - radius)
            if i == 0 or torch.all(
                (state[:i, :2] - point).norm(dim=-1) > state[:i, 4] + radius + 0.01
            ):
                break
        else:
            raise RuntimeError("could not place nonoverlapping objects")
        state[i, :2] = point
        state[i, 2:4] = torch.randn(2, generator=generator) * speed
        state[i, 4:] = torch.tensor([radius, mass])
    return state, mask


def make_dataset(
    split: str,
    scenes: int,
    horizon: int,
    seed: int,
    max_objects: int = 8,
    physics: Physics = Physics(),
) -> dict[str, torch.Tensor]:
    """Independent trajectory splits, with IDs that encode their seed domains."""
    if (
        split not in SPLITS
        or not isinstance(scenes, int)
        or not isinstance(horizon, int)
        or not isinstance(seed, int)
        or not isinstance(max_objects, int)
        or scenes < 1
        or horizon < 1
        or not 5 <= max_objects <= 16
        or not isinstance(physics, Physics)
    ):
        raise ValueError(
            "invalid split, scene count, horizon, or object capacity (minimum 5)"
        )
    data_seed = seed + OFFSETS[split]
    gen = torch.Generator().manual_seed(data_seed)
    states, masks = [], []
    for _ in range(scenes):
        n = (
            int(torch.randint(5, max_objects + 1, (), generator=gen))
            if split == "more_objects"
            else int(torch.randint(2, 5, (), generator=gen))
        )
        state, mask = sample_scene(
            gen,
            n,
            max_objects,
            split == "composition",
            0.8 if split == "fast" else 0.45,
        )
        states.append(state)
        masks.append(mask)
    current, mask = torch.stack(states), torch.stack(masks)
    # Piecewise constant control permits both free motion and sustained interventions.
    chunks = (
        torch.randn(scenes, (horizon + 3) // 4, max_objects, 2, generator=gen) * 0.3
    )
    active = (
        torch.rand(scenes, (horizon + 3) // 4, max_objects, 1, generator=gen) > 0.65
    )
    actions = (chunks * active).repeat_interleave(4, dim=1)[:, :horizon] * mask[
        :, None, :, None
    ]
    trajectory = [current]
    for t in range(horizon):
        current = simulate(current, actions[:, t], mask, physics)
        trajectory.append(current)
    return {
        "states": torch.stack(trajectory, dim=1),
        "actions": actions,
        "mask": mask,
        "ids": torch.arange(scenes, dtype=torch.int64) + data_seed * 1_000_000,
    }
