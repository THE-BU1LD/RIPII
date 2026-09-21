from __future__ import annotations

import math

import torch
from torch import nn

from ..models.quantizer import HierarchicalVectorQuantizer
from .multiresolution import (
    ConditionalMultiresolutionDynamicsV2,
    ConservativeMultiresolutionDynamics,
)

VARIANTS = (
    "mlp",
    "graph",
    "transformer",
    "global_pool",
    "multiscale",
    "equivariant",
    "ripii_mr",
)

# Experimental variants are explicitly selectable but excluded from the
# historical default benchmark grid above.  This prevents a new mechanism from
# silently changing frozen v1 comparisons.
EXPERIMENTAL_VARIANTS = ("ripii_mr_v2",)
MODEL_VARIANTS = VARIANTS + EXPERIMENTAL_VARIANTS
CONTINUOUS_DYNAMICS_VARIANTS = {"equivariant", "ripii_mr", "ripii_mr_v2"}


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
        self.half_levels = (levels - 1) / 2
        self.levels = levels
        self.last_codes: torch.Tensor | None = None

    def forward(self, h):
        bounded = torch.tanh(self.encode(h)) * self.half_levels
        rounded = bounded.round()
        self.last_codes = rounded.detach()
        discrete = bounded + (rounded - bounded).detach()
        return self.decode(discrete / self.half_levels)


class Interaction(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.message = mlp(2 * hidden + 5, hidden, hidden)
        self.update = mlp(2 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, position, velocity, mask, local: bool = True):
        n = h.shape[1]
        edges = mask.unsqueeze(2) & mask.unsqueeze(1)
        edges = edges & ~torch.eye(n, device=h.device, dtype=torch.bool)
        if local:
            edges = edges & (torch.cdist(position, position) < 0.6)
        batch, receiver, sender = edges.nonzero(as_tuple=True)
        aggregate = torch.zeros_like(h)
        if batch.numel():
            relative = position[batch, receiver] - position[batch, sender]
            dv = velocity[batch, receiver] - velocity[batch, sender]
            distance = relative.norm(dim=-1, keepdim=True)
            messages = self.message(
                torch.cat(
                    [
                        h[batch, receiver],
                        h[batch, sender],
                        relative,
                        dv,
                        distance,
                    ],
                    -1,
                )
            )
            aggregate.index_put_((batch, receiver), messages, accumulate=True)
        # Sum preserves interaction strength when scene size changes.
        return self.norm(
            h + self.update(torch.cat([h, aggregate], -1))
        ) * mask.unsqueeze(-1)


class EquivariantDynamics(nn.Module):
    """E(2)-equivariant object update built only from scalar invariants and vectors."""

    def __init__(self, hidden: int):
        super().__init__()
        self.node = mlp(5, hidden, hidden)
        self.pair = mlp(2 * hidden + 3, hidden, hidden + 2)
        self.update = mlp(2 * hidden, hidden, hidden)
        self.action_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.vector_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, state, action, mask, dt):
        live = mask.unsqueeze(-1)
        position, velocity = state[..., :2], state[..., 2:4]
        speed2 = velocity.square().sum(-1, keepdim=True)
        action2 = action.square().sum(-1, keepdim=True)
        velocity_action = (velocity * action).sum(-1, keepdim=True)
        scalars = torch.cat([speed2, action2, velocity_action, state[..., 4:]], -1)
        h = self.node(scalars) * live
        n = state.shape[1]
        left = h.unsqueeze(2).expand(-1, -1, n, -1)
        right = h.unsqueeze(1).expand(-1, n, -1, -1)
        relative = position.unsqueeze(2) - position.unsqueeze(1)
        relative_velocity = velocity.unsqueeze(2) - velocity.unsqueeze(1)
        invariants = torch.cat(
            [
                relative.square().sum(-1, keepdim=True),
                relative_velocity.square().sum(-1, keepdim=True),
                (relative * relative_velocity).sum(-1, keepdim=True),
            ],
            -1,
        )
        pair = self.pair(torch.cat([left, right, invariants], -1))
        edges = mask.unsqueeze(2) & mask.unsqueeze(1)
        edges = edges & ~torch.eye(n, dtype=torch.bool, device=state.device)
        scalar_messages, coefficients = pair[..., :-2], pair[..., -2:].tanh()
        aggregate = (scalar_messages * edges.unsqueeze(-1)).sum(2)
        h = self.update(torch.cat([h, aggregate], -1)) * live
        pair_vector = (
            coefficients[..., :1] * relative
            + coefficients[..., 1:] * relative_velocity
        )
        pair_vector = (pair_vector * edges.unsqueeze(-1)).sum(2)
        vector = pair_vector * self.vector_gate(h) + (
            action / state[..., 5:6].clamp_min(0.1)
        ) * self.action_gate(h)
        # Radial normalization is equivariant; component-wise clipping would not be.
        delta_velocity = 0.5 * vector / (1.0 + vector.norm(dim=-1, keepdim=True))
        next_velocity = velocity + delta_velocity * live
        next_position = position + dt * next_velocity * live
        return torch.cat([next_position, next_velocity, state[..., 4:]], -1) * live


class WorldModel(nn.Module):
    """Action-conditioned object dynamics; every variant uses the same kinematic prior.

    Positions, velocities, radius, mass, and force are the only inputs. No simulator
    contact calculations or future targets occur in this forward path.
    """

    def __init__(
        self,
        variant="graph",
        hidden=64,
        max_objects=8,
        dt=0.05,
        bottleneck="continuous",
        groups=4,
    ):
        super().__init__()
        if (
            variant not in MODEL_VARIANTS
            or bottleneck not in {"continuous", "fsq", "vq"}
        ):
            raise ValueError("unknown model variant or bottleneck")
        if variant in CONTINUOUS_DYNAMICS_VARIANTS and bottleneck != "continuous":
            raise ValueError(
                "equivariant dynamics variants support only a continuous scalar path"
            )
        if (
            not isinstance(hidden, int)
            or isinstance(hidden, bool)
            or not isinstance(max_objects, int)
            or isinstance(max_objects, bool)
            or not isinstance(groups, int)
            or isinstance(groups, bool)
            or hidden < 8
            or hidden % 4
            or max_objects < 5
            or not 2 <= groups <= max_objects
            or not isinstance(dt, (int, float))
            or isinstance(dt, bool)
            or not math.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError(
                "invalid width, object capacity, group count, or time step"
            )
        self.spec = {
            "variant": variant,
            "hidden": hidden,
            "max_objects": max_objects,
            "dt": dt,
            "bottleneck": bottleneck,
            "groups": groups,
        }
        self.variant, self.dt, self.max_objects = variant, dt, max_objects
        if variant == "equivariant":
            self.equivariant = EquivariantDynamics(hidden)
        elif variant == "ripii_mr":
            self.multiresolution = ConservativeMultiresolutionDynamics(hidden, groups)
        elif variant == "ripii_mr_v2":
            self.multiresolution = ConditionalMultiresolutionDynamicsV2(hidden, groups)
        else:
            self.encoder = mlp(8, hidden, hidden)
            if variant == "mlp":
                self.flat = mlp(max_objects * 9, hidden, max_objects * hidden)
            elif variant == "transformer":
                layer = nn.TransformerEncoderLayer(
                    hidden,
                    4,
                    hidden * 2,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                self.attention = nn.TransformerEncoder(
                    layer, 2, enable_nested_tensor=False
                )
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
        self.quantizer: FSQ | HierarchicalVectorQuantizer | None = None
        if bottleneck == "fsq":
            self.quantizer = FSQ(hidden)
        elif bottleneck == "vq":
            self.quantizer = HierarchicalVectorQuantizer(16, 16, hidden)
        if variant not in CONTINUOUS_DYNAMICS_VARIANTS:
            self.head = mlp(hidden, hidden, 4)
            head_output = self.head[-1]
            if not isinstance(head_output, nn.Linear):
                raise TypeError("world model output layer must be linear")
            nn.init.zeros_(head_output.weight)
            nn.init.zeros_(head_output.bias)
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
        if self.variant == "equivariant":
            self.aux_loss = state.new_zeros(())
            result = self.equivariant(state, action, mask, self.dt)
            if not torch.isfinite(result).all():
                raise FloatingPointError("non-finite world-model output")
            return result
        if self.variant in {"ripii_mr", "ripii_mr_v2"}:
            self.aux_loss = state.new_zeros(())
            result = self.multiresolution(state, action, mask, self.dt)
            self.last_assignments = self.multiresolution.last_assignments
            if not torch.isfinite(result).all():
                raise FloatingPointError("non-finite world-model output")
            return result
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
            quantizer = self.quantizer
            if not isinstance(quantizer, FSQ):
                raise RuntimeError("FSQ bottleneck is missing its quantizer")
            h = quantizer(h) * live
        elif self.bottleneck == "vq":
            quantizer = self.quantizer
            if not isinstance(quantizer, HierarchicalVectorQuantizer):
                raise RuntimeError("VQ bottleneck is missing its quantizer")
            # Padded slots must never contribute to codebook training or usage.
            quantized, stats = quantizer(h[mask])
            h = h.clone()
            h[mask] = quantized
            self.aux_loss = (
                stats["vq_commit"] + stats["vq_code"] + stats["vq_balance"]
            )
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
        if self.variant in {"ripii_mr", "ripii_mr_v2"}:
            result.update(self.multiresolution.last_diagnostics)
        if self.last_assignments is not None:
            assignment_mask = mask
            if self.variant == "ripii_mr_v2":
                multiresolution = self.multiresolution
                if not isinstance(multiresolution, ConditionalMultiresolutionDynamicsV2):
                    raise RuntimeError("ripii_mr_v2 is missing its conditional dynamics")
                routed = multiresolution.last_routed_scenes
                if routed is not None:
                    assignment_mask = assignment_mask & routed.unsqueeze(-1)
            active = self.last_assignments[assignment_mask]
            if not active.numel():
                return result
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
        if self.bottleneck == "fsq":
            quantizer = self.quantizer
            if not isinstance(quantizer, FSQ):
                raise RuntimeError("FSQ bottleneck is missing its quantizer")
            if quantizer.last_codes is None:
                return result
            active_codes = quantizer.last_codes[mask].to(torch.int64)
            utilization, effective = [], []
            for dimension in range(active_codes.shape[-1]):
                indices = active_codes[:, dimension] + int(quantizer.half_levels)
                counts = torch.bincount(
                    indices, minlength=quantizer.levels
                ).float()
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
