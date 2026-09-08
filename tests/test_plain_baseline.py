from __future__ import annotations

import torch

from ripii.models.baselines import PlainAutoencoder, match_plain_autoencoder_capacity
from ripii.models.factory import build_model
from ripii.utils.ablation import apply_mode
from ripii.utils.config import load_config


def test_plain_baseline_has_no_structured_modules_and_trains() -> None:
    cfg = apply_mode(load_config("configs/mechanism_smoke.yaml"), "plain_ae")
    model = build_model(cfg)
    assert isinstance(model, PlainAutoencoder)
    for absent in ("graph", "quantizer", "action", "renorm", "fusion"):
        assert not hasattr(model, absent)

    x = torch.randn(4, cfg.input_dim)
    losses = model.losses({"x": x}, vars(cfg.loss_weights))
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_plain_baseline_config_disables_every_ripii_mechanism() -> None:
    cfg = apply_mode(load_config("configs/mechanism_smoke.yaml"), "plain_ae")
    assert cfg.model_variant == "plain_ae"
    assert not any(
        (
            cfg.use_projective,
            cfg.use_graph,
            cfg.use_quantizer,
            cfg.use_action,
            cfg.use_spectral_loss,
            cfg.use_equivariance_loss,
            cfg.use_identity_loss,
            cfg.use_depth_loss,
        )
    )
    assert cfg.loss_weights.recon == 1.0
    assert cfg.loss_weights.kl >= 0.0


def test_plain_baseline_can_be_parameter_matched_without_padding() -> None:
    cfg = load_config("configs/mechanism_smoke.yaml")
    target = sum(p.numel() for p in build_model(cfg).parameters() if p.requires_grad)
    hidden_dim, count, error = match_plain_autoencoder_capacity(
        input_dim=cfg.input_dim,
        latent_dim=cfg.latent_dim,
        target_trainable_params=target,
        reference_hidden_dim=cfg.hidden_dim,
    )
    assert hidden_dim > 0
    assert count == sum(
        p.numel()
        for p in PlainAutoencoder(cfg.input_dim, cfg.latent_dim, hidden_dim).parameters()
        if p.requires_grad
    )
    assert error <= 0.02
