"""Force-based successor specified in ``closure_study_v1/SPECIFICATION.md``.

This module is independent of the historical WorldModel/MR implementations.
Inputs use float32 or float64; the study uses float32 and mathematical checks
use float64. Low precision is deliberately rejected because its underflow can
invalidate the specified geometric denominators. Finite padding is discarded
before arithmetic, and live physical masses are never clamped.

Checkpoint schema: ``format='ripii-closure-v1'``, the complete ``asdict(config)``
mapping, and ``model_state=model.state_dict()`` are required. Additional study
metadata is retained. A nontrainable uint8 SHA-256 configuration signature in
the state dictionary binds same-shaped weights to their variant, integration,
and physical semantics; it is a compatibility check, not authentication.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_KINDS = frozenset({"local", "global", "hierarchy", "mean", "law", "prior"})
_DTYPES = (torch.float32, torch.float64)
_FORMAT = "ripii-closure-v1"
_CAPACITY = 8


@dataclass(frozen=True)
class ClosureConfig:
    kind: str = "local"
    closure: bool = True
    substeps: int = 4
    hidden: int = 16
    groups: int = 4
    memberships: int = 2
    local_radius: float = 0.6
    force_scale: float = 20.0
    dt: float = 0.05
    drag: float = 0.06
    stiffness: float = 100.0
    damping: float = 1.5
    learned_coupling_initial: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise ValueError("unknown closure model kind")
        if type(self.closure) is not bool:
            raise TypeError("closure must be boolean")
        integers = (self.substeps, self.hidden, self.groups, self.memberships)
        if any(type(value) is not int for value in integers):
            raise TypeError("substeps, hidden, groups, and memberships must be integers")
        if (
            self.substeps < 1
            or self.hidden < 1
            or not 1 <= self.memberships <= self.groups <= _CAPACITY
        ):
            raise ValueError("invalid substeps, hidden width, or group configuration")
        positive = (
            self.local_radius,
            self.force_scale,
            self.dt,
            self.learned_coupling_initial,
        )
        nonnegative = (self.drag, self.stiffness, self.damping)
        numeric = (*positive, *nonnegative)
        if any(type(value) not in (int, float) for value in numeric):
            raise TypeError("physical and force parameters must be real numbers")
        try:
            finite = all(math.isfinite(value) for value in numeric)
        except OverflowError:
            finite = False
        if (
            not finite
            or any(value <= 0 for value in positive)
            or any(value < 0 for value in nonnegative)
        ):
            raise ValueError("invalid physical or force parameters")


_DEFAULT_CONFIG = ClosureConfig()


def _configuration_signature(config: ClosureConfig) -> torch.Tensor:
    encoded = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


def _validate_inputs(
    state: torch.Tensor,
    action: torch.Tensor,
    mask: torch.Tensor,
    *,
    sequence: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not all(isinstance(value, torch.Tensor) for value in (state, action, mask)):
        raise TypeError("state, action, and mask must be tensors")
    if any(value.layout != torch.strided for value in (state, action, mask)):
        raise TypeError("state, action, and mask must be dense tensors")
    if state.ndim != 3 or state.shape[-1] != 6:
        raise ValueError("state must have shape [batch, objects, 6]")
    batch, objects = state.shape[:2]
    if batch < 1 or not 1 <= objects <= _CAPACITY:
        raise ValueError("require a nonempty batch and 1 to 8 object slots")
    if mask.shape != state.shape[:2]:
        raise ValueError("mask must have shape [batch, objects]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if sequence:
        if (
            action.ndim != 4
            or action.shape[0] != batch
            or action.shape[2:] != (objects, 2)
        ):
            raise ValueError("actions must have shape [batch, horizon, objects, 2]")
    elif action.shape != (batch, objects, 2):
        raise ValueError("action must have shape [batch, objects, 2]")
    if state.dtype not in _DTYPES or action.dtype not in _DTYPES:
        raise TypeError("state and action must use float32 or float64")
    if action.dtype != state.dtype:
        raise TypeError("state and action must share a dtype")
    if state.device != action.device or state.device != mask.device:
        raise ValueError("state, action, and mask must share a device")
    if state.device.type == "meta":
        raise ValueError("state, action, and mask must contain materialized data")
    if not torch.isfinite(state).all() or not torch.isfinite(action).all():
        raise FloatingPointError("non-finite closure input")
    if not mask.any(dim=1).all():
        raise ValueError("every scene must contain at least one live object")
    if (state[..., 4:][mask] <= 0).any():
        raise ValueError("live objects require positive radius and mass")
    clean_state = state.masked_fill(~mask.unsqueeze(-1), 0)
    action_mask = mask[:, None, :, None] if sequence else mask.unsqueeze(-1)
    clean_action = action.masked_fill(~action_mask, 0)
    return clean_state, clean_action


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


class CentralPairForce(nn.Module):
    """Symmetric scalar edge messages applied as antisymmetric central forces."""

    def __init__(self, hidden: int, force_scale: float):
        super().__init__()
        self.net = _mlp(2 * hidden + 3, hidden, 1)
        self.force_scale = force_scale
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        features: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        mask: torch.Tensor,
        radius: float | None = None,
    ) -> torch.Tensor:
        i, j = torch.triu_indices(
            position.shape[1], position.shape[1], offset=1, device=position.device
        )
        delta = position[:, i] - position[:, j]
        relative_velocity = velocity[:, i] - velocity[:, j]
        squared_distance = delta.square().sum(-1, keepdim=True).clamp_min(1e-12)
        zi, zj = features[:, i], features[:, j]
        pair_input = torch.cat(
            (
                zi + zj,
                zi * zj,
                squared_distance,
                relative_velocity.square().sum(-1, keepdim=True),
                (delta * relative_velocity).sum(-1, keepdim=True),
            ),
            dim=-1,
        )
        strength = self.force_scale * torch.tanh(self.net(pair_input))
        valid = mask[:, i] & mask[:, j]
        if radius is not None:
            valid = valid & (delta.square().sum(-1) < radius * radius)
        pair_force = (strength * delta / squared_distance.sqrt()).masked_fill(
            ~valid.unsqueeze(-1), 0
        )
        force = torch.zeros_like(position).index_add(1, i, pair_force)
        return force.index_add(1, j, -pair_force)


@dataclass
class CoarseBatch:
    features: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    mass: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor
    shares: torch.Tensor


class ClosureCoarsener(nn.Module):
    """Geometric partition with actual occupied masses and safe inactive rows.

    Inputs have already passed the model's contract and padding sanitation.
    Exact geometric ties remain index-sensitive as declared in the specification.
    """

    def __init__(self, groups: int, memberships: int):
        super().__init__()
        self.groups = groups
        self.memberships = memberships

    def forward(
        self, features: torch.Tensor, state: torch.Tensor, mask: torch.Tensor
    ) -> CoarseBatch:
        position, velocity = state[..., :2], state[..., 2:4]
        mass = state[..., 5]
        count = mask.sum(1, keepdim=True)
        center = position.sum(1, keepdim=True) / count.unsqueeze(-1)
        from_center = (position - center).square().sum(-1)
        scale2 = (
            from_center.masked_fill(~mask, 0).sum(1, keepdim=True) / count
        ).clamp_min(1e-4)
        batch_index = torch.arange(state.shape[0], device=state.device)
        first = from_center.masked_fill(~mask, -torch.inf).argmax(1)
        anchors = [first]
        nearest = (position - position[batch_index, first, None]).square().sum(-1)
        for _ in range(1, self.groups):
            selected = nearest.masked_fill(~mask, -torch.inf).argmax(1)
            anchors.append(selected)
            distance = (
                position - position[batch_index, selected, None]
            ).square().sum(-1)
            nearest = torch.minimum(nearest, distance)
        indices = torch.stack(anchors, dim=1)
        anchor_position = position.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
        anchor_mask = torch.arange(self.groups, device=state.device)[None, :] < count
        distance2 = (position[:, :, None] - anchor_position[:, None]).square().sum(-1)
        selection_distance = distance2.masked_fill(~anchor_mask[:, None], torch.inf)
        chosen = selection_distance.topk(
            self.memberships, dim=-1, largest=False
        ).indices
        member_mask = torch.zeros_like(distance2, dtype=torch.bool).scatter(
            -1, chosen, True
        )
        member_mask = member_mask & anchor_mask[:, None] & mask.unsqueeze(-1)
        logits = (-distance2 / (2 * scale2.unsqueeze(-1))).masked_fill(
            ~member_mask, -torch.inf
        )
        # Softmax must never see an all--inf row: masking NaN outputs afterwards
        # would still leave undefined input gradients in the backward pass.
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0)
        weights = torch.softmax(logits, dim=-1).masked_fill(~mask.unsqueeze(-1), 0)
        weighted_mass = weights * mass.unsqueeze(-1)
        coarse_mass = weighted_mass.sum(1)
        occupied = coarse_mass > 0
        denominator = torch.where(occupied, coarse_mass, torch.ones_like(coarse_mass))
        shares = weighted_mass / denominator.unsqueeze(1)
        restriction = shares.transpose(1, 2)
        return CoarseBatch(
            features=torch.bmm(restriction, features),
            position=torch.bmm(restriction, position),
            velocity=torch.bmm(restriction, velocity),
            mass=coarse_mass,
            mask=occupied,
            weights=weights,
            shares=shares,
        )


def project_internal_wrench(
    force: torch.Tensor, state: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Remove translation and rotation using positive live physical masses.

    This internal primitive assumes validated inputs, sanitizes padding itself,
    and uses safe denominators even on coincident, zero-inertia configurations.
    No such projection is applied to environment or action forces.
    """
    live = mask.unsqueeze(-1)
    position = state[..., :2].masked_fill(~live, 0)
    mass = state[..., 5:6].masked_fill(~live, 0)
    force = force.masked_fill(~live, 0)
    total_mass = mass.sum(1, keepdim=True)
    shares = mass / total_mass
    center = (mass * position).sum(1, keepdim=True) / total_mass
    relative = (position - center).masked_fill(~live, 0)
    force = force - shares * force.sum(1, keepdim=True)
    torque = (
        relative[..., 0] * force[..., 1] - relative[..., 1] * force[..., 0]
    ).sum(1, keepdim=True).unsqueeze(-1)
    inertia = (mass * relative.square().sum(-1, keepdim=True)).sum(1, keepdim=True)
    nonzero = inertia > 0
    safe_inertia = torch.where(nonzero, inertia, torch.ones_like(inertia))
    coefficient = (torque / safe_inertia).masked_fill(~nonzero, 0)
    tangent = torch.stack((-relative[..., 1], relative[..., 0]), dim=-1)
    return (force - coefficient * mass * tangent).masked_fill(~live, 0)


class ClosureModel(nn.Module):
    """Declared local/global/hierarchical force models and simpler controls."""

    def __init__(self, config: ClosureConfig = _DEFAULT_CONFIG):
        super().__init__()
        if not isinstance(config, ClosureConfig):
            raise TypeError("config must be a ClosureConfig")
        self._config = config
        self.register_buffer("_config_signature", _configuration_signature(config))
        self.encoder: nn.Sequential | None = None
        self.fine: CentralPairForce | None = None
        self.second: CentralPairForce | None = None
        self.coarsener: ClosureCoarsener | None = None
        if config.kind in {"local", "global", "hierarchy", "mean"}:
            # Construction order preserves the paired encoder/fine initialization.
            self.encoder = _mlp(6, config.hidden, config.hidden)
            self.fine = CentralPairForce(config.hidden, config.force_scale)
        if config.kind in {"global", "hierarchy"}:
            self.second = CentralPairForce(config.hidden, config.force_scale)
        if config.kind == "hierarchy":
            self.coarsener = ClosureCoarsener(config.groups, config.memberships)
        self.rho: nn.Parameter | None = None
        if config.kind in {"mean", "law"}:
            target = config.learned_coupling_initial
            inverse = target + math.log(-math.expm1(-target))
            initial = torch.tensor(inverse, dtype=torch.get_default_dtype())
            if not torch.isfinite(initial):
                raise ValueError("initial coupling is not representable in the model dtype")
            self.rho = nn.Parameter(initial)

    @property
    def config(self) -> ClosureConfig:
        return self._config

    def learned_coupling(self) -> torch.Tensor | None:
        return None if self.rho is None else F.softplus(self.rho)

    def _check_placement(self, state: torch.Tensor) -> None:
        if state.device != self._config_signature.device:
            raise ValueError("inputs and model must share a device")
        parameter = next(self.parameters(), None)
        if parameter is not None and state.dtype != parameter.dtype:
            raise TypeError("input dtype must match model parameters")

    def _external_force(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.config.closure:
            return action
        position, velocity = state[..., :2], state[..., 2:4]
        radius = state[..., 4:5]
        low, high = -1 + radius, 1 - radius
        wall_low = (
            self.config.stiffness * (low - position) - self.config.damping * velocity
        ).clamp_min(0)
        wall_high = (
            self.config.stiffness * (position - high) + self.config.damping * velocity
        ).clamp_min(0)
        return (
            action
            - self.config.drag * velocity
            + wall_low * (position < low)
            - wall_high * (position > high)
        )

    def _contact_force(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        position, velocity = state[..., :2], state[..., 2:4]
        radius = state[..., 4:5]
        delta = position.unsqueeze(-2) - position.unsqueeze(-3)
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = delta / distance
        overlap = radius.unsqueeze(-2) + radius.unsqueeze(-3) - distance
        relative_velocity = velocity.unsqueeze(-2) - velocity.unsqueeze(-3)
        closing = (relative_velocity * normal).sum(-1, keepdim=True)
        strength = (
            self.config.stiffness * overlap - self.config.damping * closing
        ).clamp_min(0)
        pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pairs = pairs & ~torch.eye(
            state.shape[1], device=state.device, dtype=torch.bool
        )
        contacts = pairs.unsqueeze(-1) & (overlap > 0)
        return (normal * strength).masked_fill(~contacts, 0).sum(-2)

    def _internal_force(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        position, velocity = state[..., :2], state[..., 2:4]
        force = torch.zeros_like(position)
        if self.encoder is not None:
            assert self.fine is not None
            count = mask.sum(1, keepdim=True).to(state.dtype) / _CAPACITY
            invariants = torch.cat(
                (
                    velocity.square().sum(-1, keepdim=True),
                    action.square().sum(-1, keepdim=True),
                    (velocity * action).sum(-1, keepdim=True),
                    state[..., 4:6],
                    count.unsqueeze(-1).expand(-1, state.shape[1], -1),
                ),
                dim=-1,
            )
            features = self.encoder(invariants).masked_fill(~mask.unsqueeze(-1), 0)
            force = self.fine(features, position, velocity, mask, self.config.local_radius)
            if self.config.kind == "global":
                assert self.second is not None
                force = force + self.second(features, position, velocity, mask)
            elif self.config.kind == "hierarchy":
                assert self.second is not None and self.coarsener is not None
                coarse = self.coarsener(features, state, mask)
                coarse_force = self.second(
                    coarse.features, coarse.position, coarse.velocity, coarse.mask
                )
                prolonged = torch.bmm(coarse.shares, coarse_force)
                force = force + project_internal_wrench(prolonged, state, mask)
                force = project_internal_wrench(force, state, mask)
        elif self.config.kind == "law":
            force = self._contact_force(state, mask)
        coupling = self.learned_coupling()
        if coupling is not None:
            count = mask.sum(1, keepdim=True).unsqueeze(-1)
            center = position.sum(1, keepdim=True) / count
            force = force - coupling * (position - center)
        return force.masked_fill(~mask.unsqueeze(-1), 0)

    def internal_force(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        state, action = _validate_inputs(state, action, mask)
        self._check_placement(state)
        force = self._internal_force(state, action, mask)
        if not torch.isfinite(force).all():
            raise FloatingPointError("non-finite internal force")
        return force

    def external_force(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        state, action = _validate_inputs(state, action, mask)
        self._check_placement(state)
        force = self._external_force(state, action).masked_fill(~mask.unsqueeze(-1), 0)
        if not torch.isfinite(force).all():
            raise FloatingPointError("non-finite external force")
        return force

    def _step(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        live = mask.unsqueeze(-1)
        properties = state[..., 4:6]
        mass = torch.where(live, state[..., 5:6], torch.ones_like(state[..., 5:6]))
        h = self.config.dt / self.config.substeps
        for _ in range(self.config.substeps):
            force = self._internal_force(state, action, mask) + self._external_force(
                state, action
            )
            velocity = state[..., 2:4] + h * force / mass
            position = state[..., :2] + h * velocity
            state = torch.cat((position, velocity, properties), dim=-1).masked_fill(
                ~live, 0
            )
        if not torch.isfinite(state).all():
            raise FloatingPointError("non-finite closure output")
        return state

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        state, action = _validate_inputs(state, action, mask)
        self._check_placement(state)
        return self._step(state, action, mask)

    def rollout(
        self, state: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        state, actions = _validate_inputs(state, actions, mask, sequence=True)
        self._check_placement(state)
        trajectory = [state]
        for action in actions.unbind(dim=1):
            state = self._step(state, action, mask)
            trajectory.append(state)
        return torch.stack(trajectory, dim=1)


def load_closure_checkpoint(path: str | Path) -> tuple[ClosureModel, dict[str, Any]]:
    """Load only the complete versioned schema, rejecting semantic relabeling.

    Extra metadata (including zero completed updates for the parameter-free
    prior) is preserved. The loader does not impose training-selection policy.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("invalid closure checkpoint serialization") from exc
    required = {"format", "config", "model_state"}
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or payload["format"] != _FORMAT
    ):
        raise ValueError("invalid closure checkpoint format or required keys")
    config_data = payload["config"]
    if not isinstance(config_data, dict) or set(config_data) != {
        field.name for field in fields(ClosureConfig)
    }:
        raise ValueError("checkpoint must contain the complete exact config schema")
    config = ClosureConfig(**config_data)
    state_dict = payload["model_state"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(key, str) for key in state_dict
    ):
        raise ValueError("model_state must be a tensor state dictionary")
    # Constructing a loader must not change the caller's paired initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = ClosureModel(config)
    expected = model.state_dict()
    if set(state_dict) != set(expected):
        raise ValueError("model_state keys do not match the configured variant")
    floating_dtypes = set()
    for name, reference in expected.items():
        value = state_dict[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.shape != reference.shape
        ):
            raise ValueError(f"invalid checkpoint tensor or shape: {name}")
        if reference.is_floating_point():
            if value.dtype not in _DTYPES or not torch.isfinite(value).all():
                raise ValueError(f"invalid checkpoint floating weights: {name}")
            floating_dtypes.add(value.dtype)
        elif value.dtype != reference.dtype or not torch.equal(value, reference):
            raise ValueError("checkpoint configuration signature does not match config")
    if len(floating_dtypes) > 1:
        raise ValueError("checkpoint weights must share a floating dtype")
    if floating_dtypes:
        model = model.to(dtype=next(iter(floating_dtypes)))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint weights are incompatible") from exc
    return model.eval(), payload
