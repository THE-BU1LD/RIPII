from __future__ import annotations

import torch

from ripii.models.ripii import RIPIIModel


def test_forward_shapes():
    model = RIPIIModel(128, 96, 192, 24, 4, 3, 3, 64, 64, 2)
    x = torch.randn(8, 128)
    t = torch.randn(8, 4)
    out = model(x, x_view=torch.randn(8, 128), transform=t)
    assert out["recon"].shape == x.shape
    assert out["quantized"].shape == (8, 64)
    assert out["nodes"].shape == (8, 4, 24)
    assert len(out["stages"]) == 4
