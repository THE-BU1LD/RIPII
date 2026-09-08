from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.metrics import entropy_from_probs, perplexity_from_probs


class HierarchicalVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_coarse: int,
        num_fine: int,
        code_dim: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()

        self.num_coarse = max(1, int(num_coarse))
        self.num_fine = max(1, int(num_fine))
        self.code_dim = int(code_dim)
        self.beta = float(beta)

        self.coarse = nn.Parameter(torch.randn(self.num_coarse, self.code_dim) * 0.02)

        self.fine = nn.Parameter(torch.randn(self.num_fine, self.code_dim) * 0.02)

    def _pairwise_distance(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        c_norm = codebook.pow(2).sum(dim=-1)
        return x_norm - 2.0 * (x @ codebook.T) + c_norm

    def _quantize(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, self.code_dim)

        dist = self._pairwise_distance(flat, codebook)

        indices = torch.argmin(dist, dim=-1)

        quantized = F.embedding(indices, codebook).view_as(x)

        probs = (
            F.one_hot(indices, num_classes=codebook.shape[0])
            .to(dtype=x.dtype)
            .mean(dim=0)
        )

        return quantized, indices, probs

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coarse_quant, _, coarse_probs = self._quantize(x, self.coarse)

        residual = x - coarse_quant

        fine_quant, _, fine_probs = self._quantize(residual, self.fine)

        quantized = coarse_quant + fine_quant

        straight_through = x + (quantized - x).detach()

        commit_loss = torch.mean((x - quantized.detach()) ** 2)

        codebook_loss = torch.mean((x.detach() - quantized) ** 2)

        coarse_usage = (coarse_probs > 0).to(dtype=x.dtype).mean()

        fine_usage = (fine_probs > 0).to(dtype=x.dtype).mean()

        stats = {
            "vq_commit": self.beta * (commit_loss + 0.5 * codebook_loss),
            "vq_code": 0.5 * commit_loss + codebook_loss,
            "vq_coarse_usage": coarse_usage,
            "vq_fine_usage": fine_usage,
            "vq_coarse_entropy": entropy_from_probs(coarse_probs.unsqueeze(0)),
            "vq_fine_entropy": entropy_from_probs(fine_probs.unsqueeze(0)),
            "vq_coarse_perplexity": perplexity_from_probs(coarse_probs.unsqueeze(0)),
            "vq_fine_perplexity": perplexity_from_probs(fine_probs.unsqueeze(0)),
            "vq_usage": 0.5 * (coarse_usage + fine_usage),
            "vq_residual_energy": residual.pow(2).mean(),
        }

        return straight_through, stats
