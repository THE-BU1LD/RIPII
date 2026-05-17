from __future__ import annotations

import torch
from torch import nn

from .layers import ResidualMLPBlock, StochasticLatentHead


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.head = StochasticLatentHead(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.output_proj(h)
        return self.head(h)