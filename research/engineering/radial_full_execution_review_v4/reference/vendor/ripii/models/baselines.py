from __future__ import annotations

import torch
from torch import nn

from ..utils.metrics import mse, sanitize
from .decoder import ResidualDecoder
from .encoder import ResidualEncoder
from .layers import kl_divergence


class PlainAutoencoder(nn.Module):
    """A deliberately unstructured VAE baseline with RIPII-compatible I/O.

    This model contains only the shared residual encoder and a direct residual
    decoder.  It has no graph, projective stack, action module, fusion gate, or
    vector quantizer.  The compatibility keys allow the existing evaluation
    pipeline to compare representations without pretending those mechanisms
    are present.
    """

    model_variant = "plain_ae"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = ResidualDecoder(latent_dim, hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_view: torch.Tensor | None = None,
        transform: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del transform
        latent, mu, logvar = self.encoder(sanitize(x))
        recon = sanitize(self.decoder(latent))
        out: dict[str, torch.Tensor] = {
            "recon": recon,
            "latent": sanitize(latent),
            "pooled": sanitize(latent),
            "structural": sanitize(latent),
            "quantized": sanitize(latent),
            "mu": sanitize(mu),
            "logvar": sanitize(logvar),
        }
        if x_view is not None:
            view_latent, _, _ = self.encoder(sanitize(x_view))
            out.update(
                {
                    "view_latent": sanitize(view_latent),
                    "view_pooled": sanitize(view_latent),
                    "view_structural": sanitize(view_latent),
                    "view_quantized": sanitize(view_latent),
                }
            )
        return out

    def losses(
        self,
        batch: dict[str, torch.Tensor],
        weights: dict[str, float],
        warmup: bool = False,
    ) -> dict[str, torch.Tensor]:
        del warmup
        out = self.forward(batch["x"])
        recon = mse(out["recon"], batch["x"])
        kl = kl_divergence(out["mu"], out["logvar"])
        total = float(weights.get("recon", weights.get("rec", 1.0))) * recon
        total = total + float(weights.get("kl", 0.0)) * kl
        return {"recon": recon, "kl": kl, "total": sanitize(total)}


def match_plain_autoencoder_capacity(
    *,
    input_dim: int,
    latent_dim: int,
    target_trainable_params: int,
    reference_hidden_dim: int,
    tolerance: float = 0.02,
) -> tuple[int, int, float]:
    """Find the closest plain-baseline width without adding dummy parameters."""
    if target_trainable_params <= 0:
        raise ValueError("target_trainable_params must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be in (0, 1)")
    upper = max(16, int(reference_hidden_dim) * 4)
    candidates = []
    for hidden_dim in range(4, upper + 1):
        model = PlainAutoencoder(input_dim, latent_dim, hidden_dim)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        relative_error = abs(count - target_trainable_params) / target_trainable_params
        candidates.append((relative_error, hidden_dim, count))
    relative_error, hidden_dim, count = min(candidates)
    if relative_error > tolerance:
        raise ValueError(
            "No plain autoencoder width satisfies the requested parameter tolerance; "
            f"best hidden_dim={hidden_dim}, params={count}, relative_error={relative_error:.4f}"
        )
    return hidden_dim, count, relative_error
