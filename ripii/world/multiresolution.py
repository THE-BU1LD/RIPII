from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, output_dim),
    )


def _upper_local_edges(
    position: torch.Tensor, mask: torch.Tensor, radius: float
) -> torch.Tensor:
    count = position.shape[1]
    distances2 = (
        position.unsqueeze(2) - position.unsqueeze(1)
    ).square().sum(dim=-1)
    upper = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=position.device),
        diagonal=1,
    )
    return (
        mask.unsqueeze(2)
        & mask.unsqueeze(1)
        & upper.unsqueeze(0)
        & (distances2 < radius**2)
    )


def _upper_knn_edges(
    position: torch.Tensor, mask: torch.Tensor, neighbors: int
) -> torch.Tensor:
    count = position.shape[1]
    if count < 2:
        return torch.zeros(
            position.shape[0], count, count, dtype=torch.bool, device=position.device
        )
    distances2 = (
        position.unsqueeze(2) - position.unsqueeze(1)
    ).square().sum(dim=-1)
    valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    identity = torch.eye(count, dtype=torch.bool, device=position.device)
    distances2 = distances2.masked_fill(~valid | identity.unsqueeze(0), float("inf"))
    selected = torch.topk(
        distances2, k=min(neighbors, count - 1), dim=-1, largest=False
    ).indices
    directed = torch.zeros_like(valid)
    directed.scatter_(2, selected, True)
    directed &= valid & ~identity.unsqueeze(0)
    undirected = directed | directed.transpose(1, 2)
    upper = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=position.device),
        diagonal=1,
    )
    return undirected & upper.unsqueeze(0)


class CentralImpulse(nn.Module):
    """Learned central pair impulses with exact pair antisymmetry."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = _mlp(2 * hidden + 3, hidden, 1)
        output_layer = self.net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("central impulse output layer must be linear")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        features: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch, left, right = edges.nonzero(as_tuple=True)
        total = torch.zeros_like(position)
        if batch.numel() == 0:
            return total, 0
        relative = position[batch, left] - position[batch, right]
        relative_velocity = velocity[batch, left] - velocity[batch, right]
        distance2 = relative.square().sum(-1, keepdim=True).clamp_min(1e-12)
        invariant = torch.cat(
            [
                features[batch, left] + features[batch, right],
                features[batch, left] * features[batch, right],
                distance2,
                relative_velocity.square().sum(-1, keepdim=True),
                (relative * relative_velocity).sum(-1, keepdim=True),
            ],
            dim=-1,
        )
        magnitude = self.net(invariant).tanh()
        impulse = magnitude * relative / distance2.sqrt()
        total.index_put_((batch, left), impulse, accumulate=True)
        total.index_put_((batch, right), -impulse, accumulate=True)
        return total, int(batch.numel())


class GeometryCoarsener(nn.Module):
    """Mass-aware geometric restriction with sparse partition-of-unity weights."""

    def __init__(self, groups: int, memberships: int = 2):
        super().__init__()
        if groups < 2 or memberships < 1 or memberships > groups:
            raise ValueError("invalid group or membership count")
        self.groups = groups
        self.memberships = memberships

    @staticmethod
    def _anchors(position: torch.Tensor, mask: torch.Tensor, groups: int):
        batch_size, _, dimensions = position.shape
        anchors = position.new_zeros(batch_size, groups, dimensions)
        anchor_mask = torch.zeros(
            batch_size, groups, dtype=torch.bool, device=position.device
        )
        for batch in range(batch_size):
            live = mask[batch].nonzero(as_tuple=False).flatten()
            if live.numel() == 0:
                continue
            live_position = position[batch, live]
            active = min(groups, live.numel())
            center = live_position.mean(0)
            first = (live_position - center).square().sum(-1).argmax()
            chosen = [int(first)]
            minimum_distance = (
                live_position - live_position[first]
            ).square().sum(-1)
            for _ in range(1, active):
                candidate = int(minimum_distance.argmax())
                chosen.append(candidate)
                candidate_distance = (
                    live_position - live_position[candidate]
                ).square().sum(-1)
                minimum_distance = torch.minimum(
                    minimum_distance, candidate_distance
                )
            anchors[batch, :active] = live_position[chosen]
            anchor_mask[batch, :active] = True
        return anchors, anchor_mask

    def forward(
        self,
        features: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        mass: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchors, coarse_mask = self._anchors(position, mask, self.groups)
        distances2 = (
            position.unsqueeze(2) - anchors.unsqueeze(1)
        ).square().sum(dim=-1)
        distances2 = distances2.masked_fill(
            ~coarse_mask.unsqueeze(1), float("inf")
        )
        nearest = torch.topk(
            distances2,
            k=min(self.memberships, self.groups),
            largest=False,
            dim=-1,
        ).indices
        support = torch.zeros_like(distances2, dtype=torch.bool)
        support.scatter_(2, nearest, True)
        support &= coarse_mask.unsqueeze(1) & mask.unsqueeze(-1)
        finite_distance = distances2.masked_fill(~support, 0)
        live_count = mask.sum(1, keepdim=True).clamp_min(1).to(position.dtype)
        center = (position * mask.unsqueeze(-1)).sum(1) / live_count
        scale2 = (
            (position - center.unsqueeze(1)).square().sum(-1) * mask
        ).sum(1, keepdim=True) / live_count
        scale2 = scale2.clamp_min(1e-4).unsqueeze(-1)
        logits = -finite_distance / (2 * scale2)
        logits = logits.masked_fill(~support, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        weighted_mass = weights * mass
        coarse_mass = weighted_mass.sum(1).clamp_min(1e-8)
        share = weighted_mass / coarse_mass.unsqueeze(1)
        coarse_position = torch.einsum("bnk,bnd->bkd", share, position)
        coarse_velocity = torch.einsum("bnk,bnd->bkd", share, velocity)
        coarse_features = torch.einsum("bnk,bnh->bkh", share, features)
        live = coarse_mask.unsqueeze(-1)
        return {
            "weights": weights,
            "share": share,
            "mask": coarse_mask,
            "mass": coarse_mass.unsqueeze(-1) * live,
            "position": coarse_position * live,
            "velocity": coarse_velocity * live,
            "features": coarse_features * live,
        }


def _remove_internal_torque(
    impulse: torch.Tensor,
    position: torch.Tensor,
    mass: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    live = mask.unsqueeze(-1).to(position.dtype)
    total_mass = (mass * live).sum(1, keepdim=True).clamp_min(1e-8)
    center = (position * mass * live).sum(1, keepdim=True) / total_mass
    relative = (position - center) * live
    torque = (
        relative[..., 0] * impulse[..., 1]
        - relative[..., 1] * impulse[..., 0]
    ).sum(1, keepdim=True)
    inertia = (mass.squeeze(-1) * relative.square().sum(-1)).sum(
        1, keepdim=True
    )
    perpendicular = torch.stack([-relative[..., 1], relative[..., 0]], dim=-1)
    coefficient = torque / inertia.clamp_min(1e-8)
    corrected = impulse - coefficient.unsqueeze(-1) * mass * perpendicular
    return corrected * live


class ConservativeMultiresolutionDynamics(nn.Module):
    """E(2)-equivariant adaptive multiresolution dynamics.

    Internal fine and coarse paths conserve linear and angular momentum by
    construction. External actions are applied separately as physical forces.
    """

    def __init__(
        self,
        hidden: int,
        groups: int,
        local_radius: float = 0.6,
        coarse_neighbors: int = 2,
        memberships: int = 2,
        routing_threshold: float = 0.5,
    ):
        super().__init__()
        if (
            hidden < 8
            or hidden % 4
            or groups < 2
            or local_radius <= 0
            or coarse_neighbors < 1
            or not 0 < routing_threshold < 1
            or not all(
                math.isfinite(value) for value in (local_radius, routing_threshold)
            )
        ):
            raise ValueError("invalid multiresolution configuration")
        self.local_radius = float(local_radius)
        self.coarse_neighbors = int(coarse_neighbors)
        self.routing_threshold = float(routing_threshold)
        self.encoder = _mlp(5, hidden, hidden)
        self.fine_impulse = CentralImpulse(hidden)
        self.coarsener = GeometryCoarsener(groups, memberships)
        self.coarse_impulse = CentralImpulse(hidden)
        self.router = _mlp(hidden + 1, hidden, 1)
        router_output = self.router[-1]
        if not isinstance(router_output, nn.Linear):
            raise TypeError("router output layer must be linear")
        nn.init.constant_(router_output.bias, math.log(3.0))
        self.last_assignments: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        live = mask.unsqueeze(-1).to(state.dtype)
        position, velocity = state[..., :2], state[..., 2:4]
        radius, mass = state[..., 4:5], state[..., 5:6]
        speed2 = velocity.square().sum(-1, keepdim=True)
        action2 = action.square().sum(-1, keepdim=True)
        velocity_action = (velocity * action).sum(-1, keepdim=True)
        scalars = torch.cat([speed2, action2, velocity_action, radius, mass], dim=-1)
        features = self.encoder(scalars) * live

        fine_edges = _upper_local_edges(position, mask, self.local_radius)
        fine_impulse, fine_edge_count = self.fine_impulse(
            features, position, velocity, fine_edges
        )

        coarse = self.coarsener(features, position, velocity, mass, mask)
        coarse_edges = _upper_knn_edges(
            coarse["position"], coarse["mask"], self.coarse_neighbors
        )
        coarse_impulse, coarse_edge_count = self.coarse_impulse(
            coarse["features"],
            coarse["position"],
            coarse["velocity"],
            coarse_edges,
        )
        prolonged = torch.einsum("bnk,bkd->bnd", coarse["share"], coarse_impulse)
        prolonged = _remove_internal_torque(prolonged, position, mass, mask)

        local_scale = fine_impulse.norm(dim=-1).sum(1, keepdim=True)
        local_scale = local_scale / mask.sum(1, keepdim=True).clamp_min(1)
        scene_features = (features * mass * live).sum(1) / (
            (mass * live).sum(1).clamp_min(1e-8)
        )
        route_probability = torch.sigmoid(
            self.router(torch.cat([scene_features, local_scale], dim=-1))
        )
        hard_route = (route_probability >= self.routing_threshold).to(state.dtype)
        route = (
            hard_route + route_probability - route_probability.detach()
            if self.training
            else hard_route
        )
        internal_impulse = fine_impulse + route.unsqueeze(-1) * prolonged
        internal_impulse = _remove_internal_torque(
            internal_impulse, position, mass, mask
        )
        total_impulse = internal_impulse + dt * action * live
        next_velocity = velocity + total_impulse / mass.clamp_min(1e-8) * live
        next_position = position + dt * next_velocity * live
        result = torch.cat([next_position, next_velocity, radius, mass], dim=-1) * live

        total_internal = internal_impulse.sum(1)
        total_mass = (mass * live).sum(1, keepdim=True).clamp_min(1e-8)
        center = (position * mass * live).sum(1, keepdim=True) / total_mass
        relative = (position - center) * live
        torque = (
            relative[..., 0] * internal_impulse[..., 1]
            - relative[..., 1] * internal_impulse[..., 0]
        ).sum(1)
        partition_error = (coarse["weights"].sum(-1)[mask] - 1).abs()
        self.last_assignments = coarse["weights"].detach()
        self.last_diagnostics = {
            "partition_unity_max_error": float(
                partition_error.max().detach() if partition_error.numel() else 0.0
            ),
            "restriction_mass_max_error": float(
                (coarse["mass"].sum(1) - (mass * live).sum(1))
                .abs()
                .max()
                .detach()
            ),
            "internal_momentum_residual_max": float(
                total_internal.norm(dim=-1).max().detach()
            ),
            "internal_torque_residual_max": float(torque.abs().max().detach()),
            "routing_probability_mean": float(route_probability.mean().detach()),
            "routing_rate": float(hard_route.mean().detach()),
            "fine_edge_count": float(fine_edge_count),
            "coarse_edge_count": float(coarse_edge_count),
        }
        if not torch.isfinite(result).all() or not all(
            math.isfinite(value) for value in self.last_diagnostics.values()
        ):
            raise FloatingPointError("non-finite multiresolution dynamics")
        return result


class ConditionalMultiresolutionDynamicsV2(ConservativeMultiresolutionDynamics):
    """Versioned conditional-compute successor to RIPII-MR v1.

    Unlike :class:`ConservativeMultiresolutionDynamics`, the routing decision is
    made from fine-path invariants before restriction or coarse interactions.
    Coarse work is then executed only for routed scenes.  The separate class is
    deliberate: retained ``ripii_mr`` checkpoints and frozen v1 semantics must
    not be reinterpreted as conditional computation.
    """

    def __init__(
        self,
        hidden: int,
        groups: int,
        local_radius: float = 0.6,
        coarse_neighbors: int = 2,
        memberships: int = 2,
        routing_threshold: float = 0.5,
    ):
        super().__init__(
            hidden,
            groups,
            local_radius,
            coarse_neighbors,
            memberships,
            routing_threshold,
        )
        self.last_routed_scenes: torch.Tensor | None = None
        self._coarse_compute_calls = 0
        self._coarse_scenes_processed = 0

    def reset_compute_counters(self) -> None:
        """Reset non-checkpointed profiler counters."""
        self._coarse_compute_calls = 0
        self._coarse_scenes_processed = 0

    def compute_counters(self) -> dict[str, int]:
        """Return cumulative coarse-path work performed by this instance."""
        return {
            "coarse_compute_calls": self._coarse_compute_calls,
            "coarse_scenes_processed": self._coarse_scenes_processed,
        }

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        live = mask.unsqueeze(-1).to(state.dtype)
        position, velocity = state[..., :2], state[..., 2:4]
        radius, mass = state[..., 4:5], state[..., 5:6]
        speed2 = velocity.square().sum(-1, keepdim=True)
        action2 = action.square().sum(-1, keepdim=True)
        velocity_action = (velocity * action).sum(-1, keepdim=True)
        scalars = torch.cat([speed2, action2, velocity_action, radius, mass], dim=-1)
        features = self.encoder(scalars) * live

        fine_edges = _upper_local_edges(position, mask, self.local_radius)
        fine_impulse, fine_edge_count = self.fine_impulse(
            features, position, velocity, fine_edges
        )

        # The router consumes only fine-path scalar invariants.  No coarsening,
        # coarse edge construction, or coarse impulse evaluation has occurred.
        local_scale = fine_impulse.norm(dim=-1).sum(1, keepdim=True)
        local_scale = local_scale / mask.sum(1, keepdim=True).clamp_min(1)
        scene_features = (features * mass * live).sum(1) / (
            (mass * live).sum(1).clamp_min(1e-8)
        )
        route_probability = torch.sigmoid(
            self.router(torch.cat([scene_features, local_scale], dim=-1))
        )
        routed_scenes = route_probability.squeeze(-1) >= self.routing_threshold
        hard_route = routed_scenes.to(state.dtype).unsqueeze(-1)

        prolonged = torch.zeros_like(position)
        assignments = state.new_zeros(
            state.shape[0], state.shape[1], self.coarsener.groups
        )
        coarse_edge_count = 0
        coarse_group_count = 0
        partition_error_max = 0.0
        restriction_mass_error_max = 0.0
        selected = routed_scenes.nonzero(as_tuple=False).flatten()
        if selected.numel():
            selected_features = features.index_select(0, selected)
            selected_position = position.index_select(0, selected)
            selected_velocity = velocity.index_select(0, selected)
            selected_mass = mass.index_select(0, selected)
            selected_mask = mask.index_select(0, selected)
            coarse = self.coarsener(
                selected_features,
                selected_position,
                selected_velocity,
                selected_mass,
                selected_mask,
            )
            coarse_edges = _upper_knn_edges(
                coarse["position"], coarse["mask"], self.coarse_neighbors
            )
            coarse_impulse, coarse_edge_count = self.coarse_impulse(
                coarse["features"],
                coarse["position"],
                coarse["velocity"],
                coarse_edges,
            )
            selected_prolonged = torch.einsum(
                "bnk,bkd->bnd", coarse["share"], coarse_impulse
            )
            selected_prolonged = _remove_internal_torque(
                selected_prolonged,
                selected_position,
                selected_mass,
                selected_mask,
            )
            prolonged = prolonged.index_copy(0, selected, selected_prolonged)
            assignments = assignments.index_copy(0, selected, coarse["weights"])
            partition_error = (
                coarse["weights"].sum(-1)[selected_mask] - 1
            ).abs()
            partition_error_max = float(
                partition_error.max().detach() if partition_error.numel() else 0.0
            )
            selected_live = selected_mask.unsqueeze(-1).to(state.dtype)
            restriction_mass_error_max = float(
                (
                    coarse["mass"].sum(1)
                    - (selected_mass * selected_live).sum(1)
                )
                .abs()
                .max()
                .detach()
            )
            coarse_group_count = int(coarse["mask"].sum().detach())
            self._coarse_compute_calls += 1
            self._coarse_scenes_processed += int(selected.numel())

        route = (
            hard_route + route_probability - route_probability.detach()
            if self.training
            else hard_route
        )
        internal_impulse = fine_impulse + route.unsqueeze(-1) * prolonged
        internal_impulse = _remove_internal_torque(
            internal_impulse, position, mass, mask
        )
        total_impulse = internal_impulse + dt * action * live
        next_velocity = velocity + total_impulse / mass.clamp_min(1e-8) * live
        next_position = position + dt * next_velocity * live
        result = torch.cat([next_position, next_velocity, radius, mass], dim=-1) * live

        total_internal = internal_impulse.sum(1)
        total_mass = (mass * live).sum(1, keepdim=True).clamp_min(1e-8)
        center = (position * mass * live).sum(1, keepdim=True) / total_mass
        relative = (position - center) * live
        torque = (
            relative[..., 0] * internal_impulse[..., 1]
            - relative[..., 1] * internal_impulse[..., 0]
        ).sum(1)
        counters = self.compute_counters()
        self.last_routed_scenes = routed_scenes.detach()
        self.last_assignments = assignments.detach()
        self.last_diagnostics = {
            "partition_unity_max_error": partition_error_max,
            "restriction_mass_max_error": restriction_mass_error_max,
            "internal_momentum_residual_max": float(
                total_internal.norm(dim=-1).max().detach()
            ),
            "internal_torque_residual_max": float(torque.abs().max().detach()),
            "routing_probability_mean": float(route_probability.mean().detach()),
            "routing_rate": float(hard_route.mean().detach()),
            "routed_scene_count": float(selected.numel()),
            "fine_edge_count": float(fine_edge_count),
            "coarse_edge_count": float(coarse_edge_count),
            "coarse_group_count": float(coarse_group_count),
            "coarse_compute_calls": float(bool(selected.numel())),
            "coarse_scenes_processed": float(selected.numel()),
            "coarse_compute_calls_total": float(counters["coarse_compute_calls"]),
            "coarse_scenes_processed_total": float(
                counters["coarse_scenes_processed"]
            ),
        }
        if not torch.isfinite(result).all() or not all(
            math.isfinite(value) for value in self.last_diagnostics.values()
        ):
            raise FloatingPointError("non-finite conditional multiresolution dynamics")
        return result
