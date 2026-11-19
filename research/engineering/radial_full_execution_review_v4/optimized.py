"""Static-shape execution candidate; qualification is required before use.

The reference implementation remains authoritative. Public validation stays at
the call boundary; CUDA capture uses the unchecked, finite-input kernel only.
No architecture, parameter names, force law, or objective is changed.
"""

import copy
import importlib.util
import sys
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"
if not REFERENCE.exists():
    REFERENCE = ROOT.parents[1] / "successors/radial_adaptation_v1"
sys.path.insert(0, str(REFERENCE / "vendor"))
spec = importlib.util.spec_from_file_location(
    "radial_execution_reference",
    REFERENCE / "vendor/research/successors/simple_graph_v3/models.py",
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def dense_interaction(self, h, position, velocity, mask, local=True):
    """Evaluate all N^2 messages, mask absent edges, then sum senders."""
    n = h.shape[1]
    edges = mask.unsqueeze(2) & mask.unsqueeze(1)
    edges = edges & ~torch.eye(n, dtype=torch.bool, device=h.device)
    if local:
        edges = edges & (torch.cdist(position, position) < 0.6)
    relative = position.unsqueeze(2) - position.unsqueeze(1)
    dv = velocity.unsqueeze(2) - velocity.unsqueeze(1)
    inputs = torch.cat(
        (
            h.unsqueeze(2).expand(-1, -1, n, -1),
            h.unsqueeze(1).expand(-1, n, -1, -1),
            relative,
            dv,
            relative.norm(dim=-1, keepdim=True),
        ),
        -1,
    )
    messages = self.message(inputs)
    aggregate = torch.where(edges[..., None], messages, 0.0).sum(2)
    return self.norm(h + self.update(torch.cat((h, aggregate), -1))) * mask[..., None]


def world_forward(self, state, action, mask):
    live = mask.unsqueeze(-1)
    h = self.encoder(torch.cat((state, action), -1) * live) * live
    if self.variant == "transformer":
        h = self.attention(h, src_key_padding_mask=~mask) * live
    else:
        h = self.local(h, state[..., :2], state[..., 2:4], mask)
        if self.variant == "global_pool":
            count = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
            pooled = (h * live).sum(1, keepdim=True) / count
            context = pooled.expand(-1, self.max_objects, -1)
            h = h + self.fusion(torch.cat((h, context), -1)) * live
        h = self.refine(h, state[..., :2], state[..., 2:4], mask)
    delta = self.head(h).tanh()
    velocity = state[..., 2:4] + 0.5 * delta[..., 2:4]
    position = state[..., :2] + self.dt * velocity + 0.05 * delta[..., :2]
    return torch.cat((position, velocity, state[..., 4:]), -1) * live


def radial_step(self, state):
    position, velocity = state[..., :2], state[..., 2:4]
    relative = position[:, self.receiver] - position[:, self.sender]
    relative_velocity = velocity[:, self.receiver] - velocity[:, self.sender]
    distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    radial_speed = (relative * relative_velocity).sum(-1, keepdim=True) / distance
    instant = torch.cat(
        (
            distance,
            distance.reciprocal().clamp_max(100.0),
            radial_speed,
            relative_velocity.norm(dim=-1, keepdim=True),
        ),
        -1,
    )
    expert_input = torch.cat((self._pair_embedding, instant), -1)
    force_a = self.force_expert_a(expert_input).tanh()
    force_b = self.force_expert_b(expert_input).tanh()
    pooled = self._pair_embedding.mean(1)
    if self.adaptive:
        stats = self._scene_statistics(distance[..., 0], radial_speed[..., 0], velocity)
        gate = 0.05 + 0.90 * self.scene_gate(torch.cat((pooled, stats), -1))
    else:
        gate = pooled.new_full((pooled.shape[0], 1), 0.5)
    pair = (gate[:, None] * force_a + (1.0 - gate[:, None]) * force_b) * 20.0
    acceleration = torch.zeros_like(position)
    acceleration.index_add_(1, self.receiver, pair * (relative / distance))
    axis = torch.cat(
        (self._axis_embedding, position[..., None], velocity[..., None]), -1
    )
    acceleration = acceleration + 20.0 * self.boundary_head(axis).squeeze(-1).tanh()
    velocity = velocity + 0.05 * acceleration
    position = position + 0.05 * velocity
    return torch.cat((position, velocity), -1)


def radial_forward(self, history, horizon):
    self._observe(history)
    state = history[:, -1]
    predictions = []
    for _ in range(horizon):
        state = self._step(state)
        predictions.append(state)
    return torch.stack(predictions, 1), state.new_zeros(())


def make_optimized(original):
    model = copy.deepcopy(original)
    if isinstance(model, reference.RadialMixtureDynamics):
        model._step = MethodType(radial_step, model)
        model.forward = MethodType(radial_forward, model)
    else:
        world = model.backbone
        if world.variant not in {"graph", "global_pool", "transformer"}:
            raise ValueError("Unsupported frozen comparator")
        world.forward = MethodType(world_forward, world)
        if world.variant != "transformer":
            world.local.forward = MethodType(dense_interaction, world.local)
            world.refine.forward = MethodType(dense_interaction, world.refine)
    return model


def checked_forward(model, history, horizon):
    if history.ndim != 4 or tuple(history.shape[1:]) != (8, 5, 4):
        raise ValueError("Expected [batch, 8, 5, 4]")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Invalid horizon")
    if not torch.isfinite(history).all():
        raise FloatingPointError("Nonfinite history")
    result = model(history, horizon)
    if not torch.isfinite(result[0]).all():
        raise FloatingPointError("Nonfinite rollout")
    return result
