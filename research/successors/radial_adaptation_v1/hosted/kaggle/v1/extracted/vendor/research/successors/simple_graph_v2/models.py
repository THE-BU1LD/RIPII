"""Adaptive local/global successor; historical model implementations stay frozen."""

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
MODELS = (
    "adaptive_context",
    "fixed_half",
    "graph",
    "simple_pool",
    "transformer",
    "nri",
)


@lru_cache(maxsize=1)
def v1_models():
    """Load the frozen v1 comparator module without shadowing this module."""
    module_name = "ripii_frozen_simple_graph_v1_models"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, V1 / "models.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen v1 model module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AdaptiveDistanceInteraction(nn.Module):
    """All-pair messages with continuous invariant edge gates."""

    def __init__(self, hidden: int):
        super().__init__()
        edge_hidden = max(16, hidden // 2)
        self.edge_gate = nn.Sequential(mlp(4, edge_hidden, 1), nn.Sigmoid())
        self.message = mlp(2 * hidden + 8, hidden, hidden)
        self.update = mlp(2 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.last_gate_mean: torch.Tensor | None = None

    def forward(self, h, position, velocity, mask):
        n = h.shape[1]
        edges = mask.unsqueeze(2) & mask.unsqueeze(1)
        edges = edges & ~torch.eye(n, dtype=torch.bool, device=h.device)
        batch, receiver, sender = edges.nonzero(as_tuple=True)
        aggregate = torch.zeros_like(h)
        if batch.numel():
            relative = position[batch, receiver] - position[batch, sender]
            relative_velocity = velocity[batch, receiver] - velocity[batch, sender]
            distance = relative.norm(dim=-1, keepdim=True)
            inverse_distance = distance.clamp_min(1e-3).reciprocal()
            radial_speed = (relative * relative_velocity).sum(
                -1, keepdim=True
            ) / distance.clamp_min(1e-3)
            speed_difference = relative_velocity.norm(dim=-1, keepdim=True)
            invariants = torch.cat(
                (distance, inverse_distance, radial_speed, speed_difference), dim=-1
            )
            gate = self.edge_gate(invariants)
            message = self.message(
                torch.cat(
                    (
                        h[batch, receiver],
                        h[batch, sender],
                        relative,
                        relative_velocity,
                        invariants,
                    ),
                    dim=-1,
                )
            )
            aggregate.index_put_((batch, receiver), gate * message, accumulate=True)
            self.last_gate_mean = gate.mean()
        else:
            self.last_gate_mean = h.new_zeros(())
        live = mask.unsqueeze(-1)
        return self.norm(h + self.update(torch.cat((h, aggregate), dim=-1))) * live


class AdaptiveContextGraph(nn.Module):
    """Distance-aware pair dynamics with an adaptive global-context residual."""

    def __init__(self, hidden: int = 80, context: int = 8, adaptive: bool = True):
        super().__init__()
        if hidden < 16 or hidden % 4:
            raise ValueError("hidden width must be a multiple of four and at least 16")
        self.hidden = hidden
        self.context = context
        self.adaptive = adaptive
        self.encoder = v1_models().HistoryEncoder(hidden, context)
        self.interaction = AdaptiveDistanceInteraction(hidden)
        self.global_residual = mlp(2 * hidden, hidden, hidden)
        self.global_norm = nn.LayerNorm(hidden)
        self.scene_gate = nn.Sequential(mlp(hidden + 4, hidden, 1), nn.Sigmoid())
        self.refine = AdaptiveDistanceInteraction(hidden)
        self.head = mlp(hidden, hidden, 4)
        output = self.head[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("unexpected prediction head")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.last_scene_gates: list[torch.Tensor] = []
        self.last_edge_gates: list[torch.Tensor] = []

    def _scene_statistics(self, position, velocity, mask):
        live = mask.unsqueeze(-1)
        count = mask.sum(1, keepdim=True).clamp_min(1).to(position.dtype)
        speed = velocity.norm(dim=-1)
        speed_mean = (speed * mask).sum(1, keepdim=True) / count
        centered = (speed - speed_mean).square() * mask
        speed_std = (centered.sum(1, keepdim=True) / count).sqrt()
        distances = torch.cdist(position, position)
        edges = mask.unsqueeze(2) & mask.unsqueeze(1)
        edges = edges & ~torch.eye(
            position.shape[1], dtype=torch.bool, device=position.device
        )
        edge_count = edges.sum((1, 2), keepdim=False).clamp_min(1).to(position.dtype)
        distance_mean = (distances * edges).sum((1, 2)) / edge_count
        distance_max = distances.masked_fill(~edges, 0).amax((1, 2))
        stats = torch.cat(
            (speed_mean, speed_std, distance_mean[:, None], distance_max[:, None]),
            dim=-1,
        )
        return stats, live, count

    def _step(self, state, mask):
        position, velocity = state[..., :2], state[..., 2:4]
        action = torch.zeros_like(position)
        features = torch.cat((state, action), dim=-1)
        h = self.encoder(features) * mask.unsqueeze(-1)
        h = self.interaction(h, position, velocity, mask)
        stats, live, count = self._scene_statistics(position, velocity, mask)
        pooled = (h * live).sum(1) / count
        if self.adaptive:
            gate = self.scene_gate(torch.cat((pooled, stats), dim=-1))
        else:
            gate = pooled.new_full((pooled.shape[0], 1), 0.5)
        context = pooled.unsqueeze(1).expand(-1, h.shape[1], -1)
        global_delta = self.global_residual(torch.cat((h, context), dim=-1))
        h = self.global_norm(h + gate.unsqueeze(1) * global_delta) * live
        h = self.refine(h, position, velocity, mask)
        delta = self.head(h).tanh()
        next_velocity = velocity + 0.5 * delta[..., 2:4]
        next_position = position + 0.05 * next_velocity + 0.05 * delta[..., :2]
        result = (
            torch.cat((next_position, next_velocity, state[..., 4:]), dim=-1) * live
        )
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite adaptive-context output")
        self.last_scene_gates.append(gate.detach())
        for interaction in (self.interaction, self.refine):
            if interaction.last_gate_mean is not None:
                self.last_edge_gates.append(interaction.last_gate_mean.detach())
        return result

    def forward(self, history, horizon: int):
        if history.ndim != 4 or history.shape[1:] != (self.context, 5, 4):
            raise ValueError(f"expected [batch, {self.context}, 5, 4] history")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not torch.isfinite(history).all():
            raise FloatingPointError("non-finite history")
        self.encoder.observe(history)
        current = history[:, -1]
        properties = torch.ones_like(current[..., :2])
        properties[..., 0] = 0.04
        state = torch.cat((current, properties), dim=-1)
        mask = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        self.last_scene_gates = []
        self.last_edge_gates = []
        predictions = []
        for _ in range(horizon):
            state = self._step(state, mask)
            predictions.append(state[..., :4])
        return torch.stack(predictions, dim=1), state.new_zeros(())

    def diagnostics(self) -> dict[str, float]:
        if not self.last_scene_gates:
            return {}
        scene = torch.cat(self.last_scene_gates)
        edges = torch.stack(self.last_edge_gates)
        return {
            "gate_mean": float(scene.mean()),
            "gate_std": float(scene.std(unbiased=False)),
            "gate_min": float(scene.min()),
            "gate_max": float(scene.max()),
            "edge_gate_mean": float(edges.mean()),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@cache
def matched_hidden_sizes(
    context: int = 8, candidate_hidden: int = 80
) -> dict[str, int]:
    target = parameter_count(AdaptiveContextGraph(candidate_hidden, context, True))
    result = {"adaptive_context": candidate_hidden, "fixed_half": candidate_hidden}
    for name in MODELS[2:]:
        choices = []
        for width in range(32, 133, 4):
            model = v1_models().make_model(name, hidden=width, context=context)
            choices.append((abs(parameter_count(model) - target), width))
        result[name] = min(choices)[1]
    return result


def make_model(name: str, hidden: int | None = None, context: int = 8):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name}")
    if name in {"adaptive_context", "fixed_half"}:
        return AdaptiveContextGraph(hidden or 80, context, name == "adaptive_context")
    if hidden is None:
        hidden = matched_hidden_sizes(context)[name]
    return v1_models().make_model(name, hidden=hidden, context=context)
