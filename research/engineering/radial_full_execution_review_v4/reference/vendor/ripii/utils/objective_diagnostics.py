from __future__ import annotations

import math

import torch


def objective_gradient_diagnostics(
    model,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict:
    """Measure raw terms and weighted gradients on one fixed batch.

    The diagnostic intentionally precedes adaptive uncertainty balancing and global
    clipping so objective scale and direction remain visible.
    """
    model.eval()
    out = model.forward(
        batch["x"],
        x_view=batch.get("x_view"),
        transform=batch.get("transform"),
    )
    # RIPII keeps term construction internal because the public optimization API
    # includes weighting and balancing. This audit intentionally observes that
    # internal pre-balancing boundary without changing the trained model source.
    terms = model._base_losses(out, batch)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    active: list[str] = []
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    rows: dict[str, dict[str, float | int]] = {}
    for name, term in terms.items():
        weight = weights.get(name, 0.0)
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or weight < 0
        ):
            raise ValueError(f"invalid objective weight: {name}")
        raw = float(term.detach().cpu())
        if not math.isfinite(raw):
            raise FloatingPointError(f"non-finite objective term: {name}")
        if weight == 0:
            continue
        weighted = term * float(weight)
        grads = torch.autograd.grad(
            weighted,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        norm_sq = sum(
            float(gradient.detach().float().square().sum().cpu())
            for gradient in grads
            if gradient is not None
        )
        norm = math.sqrt(norm_sq)
        if not math.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient norm: {name}")
        active.append(name)
        gradients[name] = grads
        rows[name] = {
            "raw_value": raw,
            "weight": float(weight),
            "weighted_value": raw * float(weight),
            "weighted_gradient_l2": norm,
            "connected_parameter_tensors": sum(
                gradient is not None for gradient in grads
            ),
        }
    if not active:
        raise ValueError("objective diagnostic requires at least one active term")

    cosine: dict[str, float | None] = {}
    for left_index, left in enumerate(active):
        for right in active[left_index + 1 :]:
            left_norm = rows[left]["weighted_gradient_l2"]
            right_norm = rows[right]["weighted_gradient_l2"]
            key = f"{left}::{right}"
            if left_norm == 0 or right_norm == 0:
                cosine[key] = None
                continue
            dot = sum(
                float((a.detach().float() * b.detach().float()).sum().cpu())
                for a, b in zip(gradients[left], gradients[right], strict=False)
                if a is not None and b is not None
            )
            value = dot / (left_norm * right_norm)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite gradient cosine: {key}")
            cosine[key] = max(-1.0, min(1.0, value))
    return {
        "boundary": (
            "raw weighted terms before adaptive uncertainty balancing and global "
            "gradient clipping; one fixed batch is diagnostic, not causal evidence"
        ),
        "active_terms": active,
        "terms": rows,
        "pairwise_gradient_cosine": cosine,
    }
