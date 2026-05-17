from __future__ import annotations

import torch
from torch import nn

from .layers import ResidualMLPBlock


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)