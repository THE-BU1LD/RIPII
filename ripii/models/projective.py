
from __future__ import annotations

import torch
from torch import nn

from ..utils.metrics import cosine_distance, principal_angle_mean


def _orthonormal_basis(raw: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, : raw.shape[1]].contiguous()


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

    def project(self, x: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        if basis is None:
            basis = self.basis()
        return (x @ basis) @ basis.T

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        basis = self.basis()
        x_n = self.norm(x)
        proj = self.project(x_n, basis)
        residual = self.residual(x_n)
        gate = torch.sigmoid(self.halt(x_n))
        mix = self.mix(x_n)
        update = proj + 0.25 * residual + 0.1 * mix
        out = self.post(x + gate * update)
        proj2 = self.project(out, basis)
        orth = torch.linalg.norm(basis.T @ basis - torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)) ** 2
        idempotence = torch.mean((proj2 - self.project(proj2, basis)) ** 2)
        projection_energy = torch.mean(proj.pow(2)) / torch.mean(x_n.pow(2)).clamp_min(1e-6)
        projection_residual = torch.mean((x_n - proj).pow(2)) / torch.mean(x_n.pow(2)).clamp_min(1e-6)
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
    def __init__(self, dim: int, num_levels: int, num_projectors: int, rank: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_levels = max(0, int(num_levels))
        self.num_projectors = max(1, int(num_projectors))
        rank = rank or max(4, dim // 3)
        self.projectors = nn.ModuleList([SubspaceProjector(dim, rank) for _ in range(self.num_levels)])
        self.mixer = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(self.num_levels)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        stages = [self.norm(z)]
        stats: dict[str, torch.Tensor] = {}
        if self.num_levels == 0:
            stats.update({
                "stack_depth": torch.zeros((), device=z.device, dtype=z.dtype),
                "stack_alignment": torch.zeros((), device=z.device, dtype=z.dtype),
                "stack_geodesic": torch.zeros((), device=z.device, dtype=z.dtype),
            })
            return stages, stats
        prev_basis = None
        depths = []
        aligns = []
        geodesics = []
        current = stages[0]
        for idx, projector in enumerate(self.projectors):
            current, proj_stats, basis = projector(current)
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
            prev_basis = basis
        stats["stack_depth"] = torch.stack(depths).mean()
        stats["stack_alignment"] = torch.stack(aligns).mean()
        stats["stack_geodesic"] = torch.stack(geodesics).mean()
        return stages, stats
