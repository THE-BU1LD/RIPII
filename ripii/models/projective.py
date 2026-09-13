from __future__ import annotations

import torch
from torch import nn

from ..utils.metrics import cosine_distance, principal_angle_mean


def _orthonormal_basis(raw: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, : raw.shape[1]].contiguous()


def _raw_basis_orthogonality(raw: torch.Tensor) -> torch.Tensor:
    """Penalize collinearity before QR makes the deployed basis orthonormal."""
    normalized = raw / raw.norm(dim=0, keepdim=True).clamp_min(1e-6)
    gram = normalized.T @ normalized
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - identity).square().mean()


class SubspaceProjector(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        rank = max(1, min(int(rank), dim))
        self.dim = int(dim)
        self.rank = rank
        self.raw_basis = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.residual = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.residual.weight)
        self.norm = nn.LayerNorm(dim)
        self.halt = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.post = nn.LayerNorm(dim)

    def basis(self) -> torch.Tensor:
        return _orthonormal_basis(self.raw_basis)

    def project(
        self, x: torch.Tensor, basis: torch.Tensor | None = None
    ) -> torch.Tensor:
        if basis is None:
            basis = self.basis()
        return (x @ basis) @ basis.T

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        basis = self.basis()
        x_n = self.norm(x)
        proj = self.project(x_n, basis)
        residual = self.residual(x_n)
        gate = torch.sigmoid(self.halt(x_n))
        mix = self.mix(x_n)
        update = proj + 0.25 * residual + 0.1 * mix
        out = self.post(x + gate * update)
        proj2 = self.project(out, basis)
        # QR guarantees that the deployed projector is orthogonal.  Regularize
        # the *raw* parameterization instead, where the objective has useful
        # gradients and prevents a poorly conditioned basis.
        orth = _raw_basis_orthogonality(self.raw_basis)
        idempotence = torch.mean((proj2 - self.project(proj2, basis)) ** 2)
        projection_energy = torch.mean(proj.pow(2)) / torch.mean(x_n.pow(2)).clamp_min(
            1e-6
        )
        projection_residual = torch.mean((x_n - proj).pow(2)) / torch.mean(
            x_n.pow(2)
        ).clamp_min(1e-6)
        alignment = cosine_distance(x_n, proj)
        active_depth = gate.mean()
        stats = {
            "orthogonality": orth,
            "idempotence": idempotence,
            "projection_energy": projection_energy,
            "projection_residual": projection_residual,
            "alignment": alignment,
            "active_depth": active_depth,
        }
        return out, stats, basis


class ProjectiveRenormStack(nn.Module):
    def __init__(
        self, dim: int, num_levels: int, num_projectors: int, rank: int | None = None
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_levels = max(0, int(num_levels))
        self.num_projectors = max(1, int(num_projectors))
        rank = rank or max(4, dim // 3)
        # A shared bank is selected independently at each hierarchy level.
        # ``num_projectors`` therefore changes both capacity and behavior,
        # rather than being a dead configuration value.
        self.projectors = nn.ModuleList(
            [
                SubspaceProjector(dim, rank)
                for _ in range(self.num_projectors if self.num_levels else 0)
            ]
        )
        self.selectors = nn.ModuleList(
            [nn.Linear(dim, self.num_projectors) for _ in range(self.num_levels)]
        )
        self.mixer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(self.num_levels)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        stages = [self.norm(z)]
        stats: dict[str, torch.Tensor] = {}
        if self.num_levels == 0:
            stats.update(
                {
                    "stack_depth": torch.zeros((), device=z.device, dtype=z.dtype),
                    "stack_alignment": torch.zeros((), device=z.device, dtype=z.dtype),
                    "stack_geodesic": torch.zeros((), device=z.device, dtype=z.dtype),
                }
            )
            return stages, stats
        prev_basis = None
        depths = []
        aligns = []
        geodesics = []
        current = stages[0]
        for idx in range(self.num_levels):
            selector_weights = torch.softmax(self.selectors[idx](current), dim=-1)
            bank_results = [projector(current) for projector in self.projectors]
            bank_outputs = torch.stack([item[0] for item in bank_results], dim=1)
            current = torch.sum(selector_weights.unsqueeze(-1) * bank_outputs, dim=1)
            mean_weights = selector_weights.mean(dim=0)
            proj_stats = {
                key: torch.sum(
                    mean_weights * torch.stack([item[1][key] for item in bank_results])
                )
                for key in bank_results[0][1]
            }
            bases = [item[2] for item in bank_results]
            mixed_raw_basis = torch.sum(
                mean_weights.view(-1, 1, 1) * torch.stack(bases), dim=0
            )
            basis = _orthonormal_basis(mixed_raw_basis)
            current = self.norm(current + 0.1 * self.mixer[idx](current))
            stages.append(current)
            for key, value in proj_stats.items():
                stats[f"renorm_{idx}_{key}"] = value
            depths.append(proj_stats["active_depth"])
            aligns.append(proj_stats["alignment"])
            if prev_basis is None:
                geodesics.append(torch.zeros_like(proj_stats["alignment"]))
            else:
                geodesics.append(principal_angle_mean(prev_basis, basis))
            stats[f"renorm_{idx}_geodesic"] = geodesics[-1]
            stats[f"renorm_{idx}_projector_entropy"] = (
                -(selector_weights * selector_weights.clamp_min(1e-9).log())
                .sum(dim=-1)
                .mean()
            )
            prev_basis = basis
        stats["stack_depth"] = torch.stack(depths).mean()
        stats["stack_alignment"] = torch.stack(aligns).mean()
        stats["stack_geodesic"] = torch.stack(geodesics).mean()
        return stages, stats
