from __future__ import annotations

import torch
from torch import nn


class LatentActionModule(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        transform_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or max(64, latent_dim * 2)

        self.latent_norm = nn.LayerNorm(latent_dim)
        self.transform_norm = nn.LayerNorm(transform_dim)

        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim + transform_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(transform_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        if transform.dim() == 1:
            transform = transform.unsqueeze(0).expand(z.shape[0], -1)

        t = self.transform_norm(transform.to(dtype=z.dtype))
        z_norm = self.latent_norm(z)

        delta = self.delta_net(torch.cat([z_norm, t], dim=-1))
        gate = self.gate_net(t)

        return z + gate * delta
