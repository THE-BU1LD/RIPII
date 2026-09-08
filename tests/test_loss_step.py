from __future__ import annotations

import torch

from ripii.models.ripii import RIPIIModel


def test_backward_step():
    model = RIPIIModel(128, 96, 192, 24, 4, 3, 3, 64, 64, 2)
    x = torch.randn(16, 128)
    batch = {"x": x, "x_view": torch.randn(16, 128), "transform": torch.randn(16, 4)}
    losses = model.losses(
        batch,
        {
            "recon": 1.0,
            "equiv": 0.8,
            "inv": 0.8,
            "scale": 0.2,
            "proj": 0.2,
            "spectral": 0.1,
            "geom": 0.1,
            "vq": 1.0,
            "node": 0.1,
            "moment": 0.05,
            "identity": 0.05,
            "kl": 0.01,
            "depth": 0.05,
        },
        warmup=False,
    )
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
