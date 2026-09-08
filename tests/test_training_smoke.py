from __future__ import annotations

import torch

from ripii.models.ripii import RIPIIModel
from ripii.utils.config import Config
from ripii.utils.training import build_dataloaders, run_epoch


def test_one_training_epoch():
    cfg = Config(
        steps=1,
        batch_size=4,
        val_batch_size=4,
        dataset_size=16,
        warmup_steps=0,
        input_dim=64,
        latent_dim=32,
        hidden_dim=64,
        node_dim=8,
        num_nodes=4,
        num_levels=2,
        num_projectors=2,
        codebook_size=16,
        fine_codebook_size=16,
        codebook_dim=16,
        graph_steps=1,
        num_classes=4,
        depth_target=0.5,
        graph_topk=2,
        use_projective=True,
        use_graph=True,
        use_quantizer=True,
        use_action=True,
    )
    train_loader, val_loader = build_dataloaders(cfg)
    model = RIPIIModel(
        cfg.input_dim,
        cfg.latent_dim,
        cfg.hidden_dim,
        cfg.node_dim,
        cfg.num_nodes,
        cfg.num_levels,
        cfg.num_projectors,
        cfg.codebook_size,
        cfg.codebook_dim,
        cfg.graph_steps,
        transform_dim=cfg.transform_dim,
        depth_target=cfg.depth_target,
        graph_topk=cfg.graph_topk,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(train_loader))
    losses = model.losses(batch, vars(cfg.loss_weights), warmup=False)
    losses["total"].backward()
    opt.step()
    metrics = run_epoch(model, val_loader, opt, cfg, "cpu", train=False)
    assert "total" in metrics
