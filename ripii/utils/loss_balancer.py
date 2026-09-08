from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class AdaptiveLossBalancer(nn.Module):
    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self.keys = list(keys)
        self.log_vars = nn.Parameter(torch.zeros(len(self.keys)))

    def forward(
        self, losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        example = next(iter(losses.values()))
        total = torch.zeros((), device=example.device, dtype=example.dtype)
        terms: dict[str, torch.Tensor] = {}
        for idx, key in enumerate(self.keys):
            if key not in losses:
                continue
            weight = float(weights.get(key, 1.0))
            # A disabled objective must contribute neither its data term nor its
            # learned uncertainty offset.  Including log_var for zero-weight
            # losses gives those parameters a gradient and can distort global
            # gradient clipping in warmups and ablations.
            if weight == 0.0:
                terms[f"balanced_{key}"] = torch.zeros_like(total)
                continue
            precision = torch.exp(-self.log_vars[idx])
            term = weight * precision * losses[key] + self.log_vars[idx]
            terms[f"balanced_{key}"] = term
            total = total + term
        terms["balanced_total"] = total
        return total, terms
