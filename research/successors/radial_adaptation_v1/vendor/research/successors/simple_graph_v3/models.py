"""Physics-structured v3 successor; all predecessor implementations stay frozen."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache, lru_cache
from pathlib import Path

import torch
from torch import nn

from ripii.world.models import mlp

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "simple_graph_v1"
V2 = ROOT.parent / "simple_graph_v2"
MODELS = (
    "radial_mixture",
    "radial_fixed",
    "adaptive_context_v2",
    "graph",
    "simple_pool",
    "transformer",
)
V2_HIDDEN = 80
OBSERVATION_DT = 0.05
MAX_ACCELERATION = 20.0


def _load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def v1_models():
    """Load the frozen v1 comparator module without shadowing this module."""
    return _load_module("ripii_frozen_simple_graph_v1_models_v3", V1 / "models.py")


@lru_cache(maxsize=1)
def v2_models():
    """Load the exact frozen v2 candidate implementation as a comparator."""
    return _load_module("ripii_frozen_simple_graph_v2_models_v3", V2 / "models.py")


class RadialMixtureDynamics(nn.Module):
    """History-conditioned central forces with a bounded scene mixture.

    Pair features are invariant to object ordering and symmetric under reversing a
    pair. Forces are applied along the current relative displacement, so every
    pair contributes equal-and-opposite acceleration. A shared per-axis boundary
    term models the simulator walls. Semi-implicit Euler makes the integration
    rule explicit and keeps the candidate far cheaper than v2's recurrent graph.
    """

    def __init__(
        self,
        hidden: int = 64,
        context: int = 8,
        adaptive: bool = True,
    ):
        super().__init__()
        if hidden < 16 or hidden % 4:
            raise ValueError("hidden width must be a multiple of four and at least 16")
        if context < 2:
            raise ValueError("context must contain at least two observations")
        self.hidden = hidden
        self.context = context
        self.adaptive = adaptive

        pair_inputs = context * 3
        axis_inputs = context * 2
        self.pair_history = mlp(pair_inputs, hidden, hidden)
        self.axis_history = mlp(axis_inputs, hidden // 2, hidden // 2)
        expert_inputs = hidden + 4
        self.force_expert_a = mlp(expert_inputs, hidden, 1)
        self.force_expert_b = mlp(expert_inputs, hidden, 1)
        self.boundary_head = mlp(hidden // 2 + 2, hidden // 2, 1)
        self.scene_gate = nn.Sequential(
            mlp(hidden + 4, hidden, 1),
            nn.Sigmoid(),
        )
        for head in (
            self.force_expert_a,
            self.force_expert_b,
            self.boundary_head,
        ):
            output = head[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("unexpected acceleration head")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

        receiver, sender = (~torch.eye(5, dtype=torch.bool)).nonzero(as_tuple=True)
        self.register_buffer("receiver", receiver)
        self.register_buffer("sender", sender)
        self._pair_embedding: torch.Tensor | None = None
        self._axis_embedding: torch.Tensor | None = None
        self.last_scene_gates: list[torch.Tensor] = []
        self.last_acceleration_norms: list[torch.Tensor] = []
        self.last_momentum_residuals: list[torch.Tensor] = []

    def _observe(self, history: torch.Tensor) -> None:
        positions = history[..., :2]
        velocities = history[..., 2:4]
        relative = positions[:, :, self.receiver] - positions[:, :, self.sender]
        relative_velocity = (
            velocities[:, :, self.receiver] - velocities[:, :, self.sender]
        )
        distance = relative.norm(dim=-1).clamp_min(1e-4)
        radial_speed = (relative * relative_velocity).sum(-1) / distance
        speed_difference = relative_velocity.norm(dim=-1)
        invariant_history = torch.stack(
            (distance, radial_speed, speed_difference), dim=-1
        )
        self._pair_embedding = self.pair_history(
            invariant_history.transpose(1, 2).flatten(2)
        )

        # The same one-dimensional wall law is shared across objects and axes.
        axis_history = torch.stack((positions, velocities), dim=-1)
        self._axis_embedding = self.axis_history(
            axis_history.permute(0, 2, 3, 1, 4).flatten(3)
        )

    @staticmethod
    def _scene_statistics(
        distance: torch.Tensor,
        radial_speed: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (
                distance.mean(1),
                distance.std(1, unbiased=False),
                radial_speed.abs().mean(1),
                speed.norm(dim=-1).mean(1),
            ),
            dim=-1,
        )

    def _step(self, state: torch.Tensor) -> torch.Tensor:
        if self._pair_embedding is None or self._axis_embedding is None:
            raise RuntimeError("Observed history must be supplied before prediction")
        position, velocity = state[..., :2], state[..., 2:4]
        relative = position[:, self.receiver] - position[:, self.sender]
        relative_velocity = velocity[:, self.receiver] - velocity[:, self.sender]
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        unit = relative / distance
        inverse_distance = distance.reciprocal().clamp_max(100.0)
        radial_speed = (relative * relative_velocity).sum(-1, keepdim=True) / distance
        speed_difference = relative_velocity.norm(dim=-1, keepdim=True)
        instant = torch.cat(
            (distance, inverse_distance, radial_speed, speed_difference), dim=-1
        )
        expert_input = torch.cat((self._pair_embedding, instant), dim=-1)
        force_a = self.force_expert_a(expert_input).tanh()
        force_b = self.force_expert_b(expert_input).tanh()
        stats = self._scene_statistics(
            distance[..., 0], radial_speed[..., 0], velocity
        )
        pooled = self._pair_embedding.mean(1)
        if self.adaptive:
            # Stay away from exact 0/1 while still permitting meaningful adaptation.
            gate = 0.05 + 0.90 * self.scene_gate(torch.cat((pooled, stats), dim=-1))
        else:
            gate = pooled.new_full((pooled.shape[0], 1), 0.5)
        scalar_force = (
            gate[:, None] * force_a + (1.0 - gate[:, None]) * force_b
        ) * MAX_ACCELERATION
        pair_acceleration = scalar_force * unit
        acceleration = torch.zeros_like(position)
        acceleration.index_add_(1, self.receiver, pair_acceleration)

        axis_input = torch.cat(
            (self._axis_embedding, position.unsqueeze(-1), velocity.unsqueeze(-1)),
            dim=-1,
        )
        boundary = self.boundary_head(axis_input).squeeze(-1).tanh()
        acceleration = acceleration + MAX_ACCELERATION * boundary
        next_velocity = velocity + OBSERVATION_DT * acceleration
        next_position = position + OBSERVATION_DT * next_velocity
        result = torch.cat((next_position, next_velocity), dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite radial-mixture output")

        momentum_residual = torch.zeros_like(position)
        momentum_residual.index_add_(1, self.receiver, pair_acceleration)
        self.last_scene_gates.append(gate.detach())
        self.last_acceleration_norms.append(acceleration.norm(dim=-1).mean().detach())
        self.last_momentum_residuals.append(
            momentum_residual.sum(1).norm(dim=-1).mean().detach()
        )
        return result

    def forward(self, history: torch.Tensor, horizon: int):
        expected = (self.context, 5, 4)
        if history.ndim != 4 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"expected [batch, {self.context}, 5, 4] history")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not torch.isfinite(history).all():
            raise FloatingPointError("non-finite history")
        self._observe(history)
        state = history[:, -1]
        self.last_scene_gates = []
        self.last_acceleration_norms = []
        self.last_momentum_residuals = []
        predictions = []
        for _ in range(horizon):
            state = self._step(state)
            predictions.append(state)
        return torch.stack(predictions, dim=1), state.new_zeros(())

    def diagnostics(self) -> dict[str, float]:
        if not self.last_scene_gates:
            return {}
        scene = torch.cat(self.last_scene_gates)
        acceleration = torch.stack(self.last_acceleration_norms)
        momentum = torch.stack(self.last_momentum_residuals)
        return {
            "gate_mean": float(scene.mean()),
            "gate_std": float(scene.std(unbiased=False)),
            "gate_min": float(scene.min()),
            "gate_max": float(scene.max()),
            "acceleration_norm_mean": float(acceleration.mean()),
            "pair_momentum_residual_max": float(momentum.max()),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@cache
def matched_hidden_sizes(
    context: int = 8, candidate_hidden: int = 64
) -> dict[str, int]:
    target = parameter_count(RadialMixtureDynamics(candidate_hidden, context, True))
    result = {
        "radial_mixture": candidate_hidden,
        "radial_fixed": candidate_hidden,
        # Exact predecessor width from the frozen v2 protocol. More capacity makes
        # this a conservative comparator instead of silently weakening v2.
        "adaptive_context_v2": V2_HIDDEN,
    }
    for name in MODELS[3:]:
        choices = []
        for width in range(32, 133, 4):
            model = v1_models().make_model(name, hidden=width, context=context)
            choices.append((abs(parameter_count(model) - target), width))
        result[name] = min(choices)[1]
    return result


def make_model(name: str, hidden: int | None = None, context: int = 8):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name}")
    if name in {"radial_mixture", "radial_fixed"}:
        return RadialMixtureDynamics(
            hidden or 64,
            context,
            adaptive=name == "radial_mixture",
        )
    if name == "adaptive_context_v2":
        if hidden is not None and hidden != V2_HIDDEN:
            raise ValueError("frozen v2 comparator must use hidden width 80")
        return v2_models().make_model("adaptive_context", V2_HIDDEN, context)
    if hidden is None:
        hidden = matched_hidden_sizes(context)[name]
    return v1_models().make_model(name, hidden=hidden, context=context)
