from __future__ import annotations

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(dim) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h * self.scale


class StochasticLatentHead(nn.Module):
    def __init__(self, dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(dim, latent_dim)
        self.logvar = nn.Linear(dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mean(x)
        logvar = self.logvar(x).clamp(-6.0, 2.0)
        std = torch.exp(0.5 * logvar)
        # Checkpoint evaluation must be deterministic. Sampling is part of the
        # variational training objective, not the evaluation function.
        z = mu + torch.randn_like(std) * std if self.training else mu
        return z, mu, logvar


class TransformEmbed(nn.Module):
    def __init__(self, transform_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden = max(32, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(transform_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, transform: torch.Tensor) -> torch.Tensor:
        return self.net(transform)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
    )
