from __future__ import annotations

import math

import torch
from torch import nn

from ..models.quantizer import HierarchicalVectorQuantizer

VARIANTS = ("mlp", "graph", "transformer", "global_pool", "multiscale")


def mlp(input_dim: int, hidden: int, output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output)
    )


class FSQ(nn.Module):
    """Fixed scalar levels with a straight-through estimator, no learned codebook."""

    def __init__(self, hidden: int, dimensions: int = 4, levels: int = 5):
        super().__init__()
        if hidden < 1 or dimensions < 1 or levels < 3 or levels % 2 == 0:
            raise ValueError("FSQ requires positive widths and an odd level count >= 3")
        self.encode = nn.Linear(hidden, dimensions)
        self.decode = nn.Linear(dimensions, hidden)
        self.half = (levels - 1) / 2
        self.levels = levels
        self.last_codes: torch.Tensor | None = None

    def forward(self, h):
        bounded = torch.tanh(self.encode(h)) * self.half
        rounded = bounded.round()
        self.last_codes = rounded.detach()
        discrete = bounded + (rounded - bounded).detach()
        return self.decode(discrete / self.half)


class Interaction(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.message = mlp(2 * hidden + 5, hidden, hidden)
        self.update = mlp(2 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, position, velocity, mask, local: bool = True):
        n = h.shape[1]
        left, right = (
            h.unsqueeze(2).expand(-1, -1, n, -1),
            h.unsqueeze(1).expand(-1, n, -1, -1),
        )
        relative = position.unsqueeze(2) - position.unsqueeze(1)
        dv = velocity.unsqueeze(2) - velocity.unsqueeze(1)
        distance = relative.norm(dim=-1, keepdim=True)
        edges = mask.unsqueeze(2) & mask.unsqueeze(1)
        edges = edges & ~torch.eye(n, device=h.device, dtype=torch.bool)
        if local:
            edges = edges & (distance.squeeze(-1) < 0.6)
        messages = self.message(torch.cat([left, right, relative, dv, distance], -1))
        # Sum preserves interaction strength when scene size changes.
        aggregate = (messages * edges.unsqueeze(-1)).sum(2)
        return self.norm(
            h + self.update(torch.cat([h, aggregate], -1))
        ) * mask.unsqueeze(-1)


class WorldModel(nn.Module):
    """Action-conditioned object dynamics; every variant uses the same kinematic prior.

    Positions, velocities, radius, mass, and force are the only inputs. No simulator
    contact calculations or future targets occur in this forward path.
    """

    def __init__(
        self,
        variant="multiscale",
        hidden=64,
        max_objects=8,
        dt=0.05,
        bottleneck="continuous",
        groups=4,
    ):
        super().__init__()
        if variant not in VARIANTS or bottleneck not in {"continuous", "fsq", "vq"}:
            raise ValueError("unknown model variant or bottleneck")
        if (
            not isinstance(hidden, int)
            or not isinstance(max_objects, int)
            or not isinstance(groups, int)
            or hidden < 8
            or hidden % 4
            or max_objects < 5
            or not 2 <= groups <= max_objects
            or not math.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError(
                "invalid width, object capacity, group count, or time step"
            )
        self.spec = dict(
            variant=variant,
            hidden=hidden,
            max_objects=max_objects,
            dt=dt,
            bottleneck=bottleneck,
            groups=groups,
        )
        self.variant, self.dt, self.max_objects = variant, dt, max_objects
        self.encoder = mlp(8, hidden, hidden)
        if variant == "mlp":
            self.flat = mlp(max_objects * 9, hidden, max_objects * hidden)
        elif variant == "transformer":
            layer = nn.TransformerEncoderLayer(
                hidden, 4, hidden * 2, dropout=0.0, activation="gelu", batch_first=True
            )
            self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        else:
            self.local = Interaction(hidden)
            self.refine = Interaction(hidden)
            if variant == "multiscale":
                self.assignment = nn.Linear(hidden, groups)
                self.coarse = Interaction(hidden)
                self.fusion = mlp(hidden * 2, hidden, hidden)
            elif variant == "global_pool":
                self.fusion = mlp(hidden * 2, hidden, hidden)
        self.bottleneck = bottleneck
        if bottleneck == "fsq":
            self.quantizer = FSQ(hidden)
        elif bottleneck == "vq":
            self.quantizer = HierarchicalVectorQuantizer(16, 16, hidden)
        self.head = mlp(hidden, hidden, 4)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.aux_loss = torch.tensor(0.0)
        self.last_assignments = None
        self.last_quantizer_stats: dict[str, torch.Tensor] = {}

    def forward(self, state, action, mask):
        if state.ndim != 3 or state.shape[1:] != (self.max_objects, 6):
            raise ValueError(f"expected [batch, {self.max_objects}, 6] state")
        if action.shape != (*state.shape[:2], 2) or mask.shape != state.shape[:2]:
            raise ValueError("invalid actions or empty scene mask")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        if not state.is_floating_point() or not action.is_floating_point():
            raise TypeError("state and action must be floating-point tensors")
        if state.device != action.device or state.device != mask.device:
            raise ValueError("state, action, and mask must share a device")
        if not mask.any(dim=1).all():
            raise ValueError("every scene must contain at least one live object")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise FloatingPointError("non-finite world-model input")
        if (state[..., 4:][mask] <= 0).any():
            raise ValueError("live objects require positive radius and mass")
        live = mask.unsqueeze(-1)
        self.last_assignments = None
        self.last_quantizer_stats = {}
        features = torch.cat([state, action], -1) * live
        h = self.encoder(features) * live
        if self.variant == "mlp":
            flat = torch.cat([features, live.to(state.dtype)], -1).flatten(1)
            h = self.flat(flat).reshape(state.shape[0], self.max_objects, -1) * live
        elif self.variant == "transformer":
            h = self.attention(h, src_key_padding_mask=~mask) * live
        else:
            h = self.local(h, state[..., :2], state[..., 2:4], mask)
            if self.variant == "multiscale":
                assignment = self.assignment(h).softmax(-1) * live
                mass = assignment.sum(1).clamp_min(1e-6)
                weights = assignment.transpose(1, 2) / mass.unsqueeze(-1)
                coarse_h = weights @ h
                coarse_p, coarse_v = weights @ state[..., :2], weights @ state[..., 2:4]
                coarse_mask = torch.ones(
                    mass.shape, dtype=torch.bool, device=state.device
                )
                coarse_h = self.coarse(
                    coarse_h, coarse_p, coarse_v, coarse_mask, local=False
                )
                h = h + self.fusion(torch.cat([h, assignment @ coarse_h], -1)) * live
                self.last_assignments = assignment.detach()
            elif self.variant == "global_pool":
                count = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
                pooled = (h * live).sum(1, keepdim=True) / count
                global_context = pooled.expand(-1, self.max_objects, -1)
                h = h + self.fusion(torch.cat([h, global_context], -1)) * live
            h = self.refine(h, state[..., :2], state[..., 2:4], mask)
        self.aux_loss = h.new_zeros(())
        if self.bottleneck == "fsq":
            h = self.quantizer(h) * live
        elif self.bottleneck == "vq":
            # Padded slots must never contribute to codebook training or usage.
            quantized, stats = self.quantizer(h[mask])
            h = h.clone()
            h[mask] = quantized
            self.aux_loss = stats["vq_commit"] + stats["vq_code"]
            self.last_quantizer_stats = {
                key: value.detach() for key, value in stats.items()
            }
        delta = self.head(h).tanh()
        velocity = state[..., 2:4] + 0.5 * delta[..., 2:4]
        position = state[..., :2] + self.dt * velocity + 0.05 * delta[..., :2]
        result = torch.cat([position, velocity, state[..., 4:]], -1) * live
        if not torch.isfinite(result).all() or not torch.isfinite(self.aux_loss).all():
            raise FloatingPointError("non-finite world-model output")
        return result

    @torch.no_grad()
    def diagnostics(self, mask: torch.Tensor) -> dict[str, float]:
        """Return bounded mechanism diagnostics for the most recent forward pass."""
        result: dict[str, float] = {}
        if self.last_assignments is not None:
            active = self.last_assignments[mask]
            eps = torch.finfo(active.dtype).eps
            per_object_entropy = -(active * active.clamp_min(eps).log()).sum(-1)
            occupancy = active.mean(0)
            occupancy = occupancy / occupancy.sum().clamp_min(eps)
            occupancy_entropy = -(occupancy * occupancy.clamp_min(eps).log()).sum()
            result.update(
                {
                    "assignment_entropy": float(per_object_entropy.mean()),
                    "assignment_normalized_entropy": float(
                        per_object_entropy.mean() / math.log(active.shape[-1])
                    ),
                    "assignment_effective_groups": float(occupancy_entropy.exp()),
                    "assignment_min_occupancy": float(occupancy.min()),
                    "assignment_max_occupancy": float(occupancy.max()),
                }
            )
        if self.bottleneck == "fsq" and self.quantizer.last_codes is not None:
            active_codes = self.quantizer.last_codes[mask].to(torch.int64)
            utilization, effective = [], []
            for dimension in range(active_codes.shape[-1]):
                indices = active_codes[:, dimension] + int(self.quantizer.half)
                counts = torch.bincount(indices, minlength=self.quantizer.levels).float()
                probabilities = counts / counts.sum().clamp_min(1)
                nonzero = probabilities > 0
                entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
                utilization.append(float(nonzero.float().mean()))
                effective.append(float(entropy.exp()))
            result.update(
                {
                    "fsq_level_utilization": sum(utilization) / len(utilization),
                    "fsq_effective_levels": sum(effective) / len(effective),
                }
            )
        if self.bottleneck == "vq":
            result.update(
                {
                    key: float(value)
                    for key, value in self.last_quantizer_stats.items()
                    if key
                    in {
                        "vq_coarse_usage",
                        "vq_fine_usage",
                        "vq_coarse_entropy",
                        "vq_fine_entropy",
                        "vq_coarse_perplexity",
                        "vq_fine_perplexity",
                        "vq_residual_energy",
                    }
                }
            )
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("non-finite world-model diagnostic")
        return result


def rollout(
    model: WorldModel, state: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if actions.ndim != 4 or actions.shape[0] != state.shape[0]:
        raise ValueError("actions must have shape [batch, time, objects, 2]")
    states = [state]
    for t in range(actions.shape[1]):
        state = model(state, actions[:, t], mask)
        states.append(state)
    return torch.stack(states, dim=1)
