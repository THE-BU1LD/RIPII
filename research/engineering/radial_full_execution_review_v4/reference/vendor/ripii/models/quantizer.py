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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, self.code_dim)

        dist = self._pairwise_distance(flat, codebook)

        indices = torch.argmin(dist, dim=-1)

        quantized = F.embedding(indices, codebook).view_as(x)

        probs = (
            F.one_hot(indices, num_classes=codebook.shape[0])
            .to(dtype=x.dtype)
            .mean(dim=0)
        )
        soft_probs = torch.softmax(-dist, dim=-1).mean(dim=0)

        return quantized, indices, probs, soft_probs

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coarse_quant, _, coarse_probs, coarse_soft_probs = self._quantize(
            x, self.coarse
        )

        residual = x - coarse_quant

        fine_quant, _, fine_probs, fine_soft_probs = self._quantize(residual, self.fine)

        quantized = coarse_quant + fine_quant

        straight_through = x + (quantized - x).detach()

        commit_loss = torch.mean((x - quantized.detach()) ** 2)

        codebook_loss = torch.mean((x.detach() - quantized) ** 2)

        coarse_usage = (coarse_probs > 0).to(dtype=x.dtype).mean()

        fine_usage = (fine_probs > 0).to(dtype=x.dtype).mean()

        # KL(batch-average soft assignments || uniform) is differentiable with
        # respect to both encoder outputs and codebooks.  Hard usage remains a
        # diagnostic and is never used as a training objective.
        coarse_balance = torch.sum(
            coarse_soft_probs
            * (
                coarse_soft_probs.clamp_min(1e-9).log()
                + torch.log(
                    torch.tensor(float(self.num_coarse), device=x.device, dtype=x.dtype)
                )
            )
        )
        fine_balance = torch.sum(
            fine_soft_probs
            * (
                fine_soft_probs.clamp_min(1e-9).log()
                + torch.log(
                    torch.tensor(float(self.num_fine), device=x.device, dtype=x.dtype)
                )
            )
        )

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
            "vq_balance": 0.5 * (coarse_balance + fine_balance),
            "vq_residual_energy": residual.pow(2).mean(),
        }

        return straight_through, stats

    @torch.no_grad()
    def revive_dead_codes(
        self, x: torch.Tensor, min_assignments: int = 1
    ) -> dict[str, int]:
        """Deterministically re-seed underused codes from poorly represented inputs.

        This is an explicit training intervention, never an evaluation side effect.
        Callers must record the schedule in the experiment configuration.
        """
        if (
            x.shape[-1] != self.code_dim
            or x.numel() == 0
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
            or not isinstance(min_assignments, int)
            or isinstance(min_assignments, bool)
            or min_assignments < 1
        ):
            raise ValueError("invalid dead-code revival input")

        def revive(values: torch.Tensor, codebook: torch.Tensor):
            flat = values.reshape(-1, self.code_dim)
            distance = self._pairwise_distance(flat, codebook)
            indices = distance.argmin(-1)
            counts = torch.bincount(indices, minlength=codebook.shape[0])
            dead = (counts < min_assignments).nonzero(as_tuple=False).flatten()
            if dead.numel():
                priority = distance.min(-1).values.argsort(descending=True)
                selected = priority[
                    torch.arange(dead.numel(), device=priority.device) % priority.numel()
                ]
                codebook[dead] = flat[selected]
            return int(dead.numel()), F.embedding(indices, codebook).view_as(values)

        coarse_reset, coarse_quantized = revive(x, self.coarse)
        fine_reset, _ = revive(x - coarse_quantized, self.fine)
        return {"coarse_codes_reset": coarse_reset, "fine_codes_reset": fine_reset}
