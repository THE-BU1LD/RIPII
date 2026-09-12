from __future__ import annotations

import torch

from ripii.models.factory import build_model
from ripii.utils.config import load_config
from ripii.utils.objective_diagnostics import objective_gradient_diagnostics


def test_objective_diagnostics_reports_real_finite_gradients() -> None:
    cfg = load_config("configs/mechanism_smoke.yaml")
    model = build_model(cfg)
    batch = {
        "x": torch.randn(4, cfg.input_dim),
        "x_view": torch.randn(4, cfg.input_dim),
        "transform": torch.randn(4, cfg.transform_dim),
    }
    weights = vars(cfg.loss_weights).copy()
    weights["vq"] = 0.0
    result = objective_gradient_diagnostics(model, batch, weights)
    assert "vq" not in result["active_terms"]
    assert result["terms"]["recon"]["weighted_gradient_l2"] > 0
    assert result["pairwise_gradient_cosine"]
    assert all(
        value is None or -1.0 <= value <= 1.0
        for value in result["pairwise_gradient_cosine"].values()
    )
