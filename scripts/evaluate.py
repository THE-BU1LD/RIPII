
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from ripii.models.ripii import RIPIIModel
from ripii.utils.config import load_config, runtime_profile
from ripii.utils.metrics import effective_rank, ridge_probe_accuracy, cosine_similarity, mse
from ripii.utils.training import build_dataloaders, load_checkpoint


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
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    cfg = runtime_profile(load_config(args.config))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    totals = {}
    pooled_features = []
    structural_features = []
    labels = []
    consistency = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses = model.losses(batch, vars(cfg.loss_weights), warmup=False)
            out = model.forward(batch["x"], x_view=batch["x_view"], transform=batch.get("transform"))
            pooled_features.append(out["pooled"].detach().cpu())
            structural_features.append(out["structural"].detach().cpu())
            labels.append(batch["label"].detach().cpu())
            consistency.append({
                "latent_cosine": float(cosine_similarity(out["pooled"], out["view_pooled"] if "view_pooled" in out else out["pooled"]).detach().cpu()),
                "structural_mse": float(mse(out["structural"], out["view_structural"] if "view_structural" in out else out["structural"]).detach().cpu()),
                "quantized_cosine": float(cosine_similarity(out["quantized"], out["view_quantized"] if "view_quantized" in out else out["quantized"]).detach().cpu()),
            })
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
                else:
                    totals[k] = totals.get(k, 0.0) + float(v)
    n = max(1, len(val_loader))
    summary = {k: v / n for k, v in totals.items()}
    pooled = torch.cat(pooled_features, dim=0)
    structural = torch.cat(structural_features, dim=0)
    labs = torch.cat(labels, dim=0)
    summary["probe_accuracy"] = ridge_probe_accuracy(pooled, labs, cfg.num_classes)
    summary["structural_probe_accuracy"] = ridge_probe_accuracy(structural, labs, cfg.num_classes)
    summary["feature_effective_rank"] = float(effective_rank(pooled).item())
    summary["structural_effective_rank"] = float(effective_rank(structural).item())
    if consistency:
        summary["latent_cosine_mean"] = sum(c["latent_cosine"] for c in consistency) / len(consistency)
        summary["structural_mse_mean"] = sum(c["structural_mse"] for c in consistency) / len(consistency)
        summary["quantized_cosine_mean"] = sum(c["quantized_cosine"] for c in consistency) / len(consistency)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
