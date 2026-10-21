from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

OBJECTIVE_SEMANTICS_VERSION = "adaptive-loss-balancer-v2-zero-weight-exact"


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
        fixed_total = torch.zeros_like(total)
        terms: dict[str, torch.Tensor] = {}
        for idx, key in enumerate(self.keys):
            if key not in losses:
                continue
            weight = float(weights.get(key, 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"loss weight must be finite and nonnegative: {key}")
            # A disabled objective must contribute neither its data term nor its
            # learned uncertainty offset.  Including log_var for zero-weight
            # losses gives those parameters a gradient and can distort global
            # gradient clipping in warmups and ablations.
            if weight == 0.0:
                terms[f"balanced_{key}"] = torch.zeros_like(total)
                terms[f"effective_weight_{key}"] = torch.zeros_like(total)
                continue
            precision = torch.exp(-self.log_vars[idx])
            term = weight * precision * losses[key] + self.log_vars[idx]
            terms[f"balanced_{key}"] = term
            terms[f"adaptive_precision_{key}"] = precision
            terms[f"effective_weight_{key}"] = weight * precision
            terms[f"log_variance_{key}"] = self.log_vars[idx]
            total = total + term
            fixed_total = fixed_total + weight * losses[key]
        terms["balanced_total"] = total
        # This value is diagnostic only: unlike balanced_total it remains comparable
        # across steps for a fixed active objective and exposes log-variance gaming.
        terms["fixed_weight_total"] = fixed_total
        return total, terms
