from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ripii.models.ripii import RIPIIModel
from ripii.utils.config import load_config, runtime_profile
from ripii.utils.training import load_checkpoint


def build_model(cfg):
    return RIPIIModel(
        input_dim=cfg.input_dim,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        node_dim=cfg.node_dim,
        num_nodes=cfg.num_nodes,
        num_levels=cfg.num_levels,
        num_projectors=cfg.num_projectors,
        codebook_size=cfg.codebook_size,
        codebook_dim=cfg.codebook_dim,
        graph_steps=cfg.graph_steps,
        transform_dim=cfg.transform_dim,
        fine_codebook_size=cfg.fine_codebook_size,
        depth_target=cfg.depth_target,
        graph_topk=cfg.graph_topk,
        use_projective=cfg.use_projective,
        use_graph=cfg.use_graph,
        use_quantizer=cfg.use_quantizer,
        use_action=cfg.use_action,
        use_spectral_loss=cfg.use_spectral_loss,
        use_equivariance_loss=cfg.use_equivariance_loss,
        use_identity_loss=cfg.use_identity_loss,
        use_depth_loss=cfg.use_depth_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    cfg = runtime_profile(load_config(args.config))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = build_model(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=str(device))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "input_dim": cfg.input_dim,
        "latent_dim": cfg.latent_dim,
        "hidden_dim": cfg.hidden_dim,
        "node_dim": cfg.node_dim,
        "num_nodes": cfg.num_nodes,
        "num_levels": cfg.num_levels,
        "num_projectors": cfg.num_projectors,
        "codebook_size": cfg.codebook_size,
        "fine_codebook_size": cfg.fine_codebook_size,
        "codebook_dim": cfg.codebook_dim,
        "graph_steps": cfg.graph_steps,
        "use_projective": cfg.use_projective,
        "use_graph": cfg.use_graph,
        "use_quantizer": cfg.use_quantizer,
        "use_action": cfg.use_action,
        "use_spectral_loss": cfg.use_spectral_loss,
        "use_equivariance_loss": cfg.use_equivariance_loss,
        "use_identity_loss": cfg.use_identity_loss,
        "use_depth_loss": cfg.use_depth_loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
