
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

import matplotlib.pyplot as plt
import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from ripii.models.ripii import RIPIIModel
from ripii.utils.config import load_config, runtime_profile
from ripii.utils.metrics import effective_rank, ridge_probe_accuracy, subspace_overlap
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


def save_line_plot(path: Path, xs, ys, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    cfg = runtime_profile(load_config(args.config))
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    rows = []
    pooled = []
    structural = []
    nodes = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward(batch["x"], x_view=batch["x_view"], transform=batch.get("transform"))
            rows.append({
                "depth": float(out["expected_depth"].detach().cpu()),
                "usage": float(out["vq_usage"].detach().cpu()),
                "perplexity": float(out["vq_coarse_perplexity"].detach().cpu()),
                "node_entropy": float(out["node_entropy"].detach().cpu()),
                "stack_alignment": float(out["stack_alignment"].detach().cpu()),
                "stack_geodesic": float(out["stack_geodesic"].detach().cpu()),
                "gate_mean": float(out["struct_gate_mean"].detach().cpu()),
            })
            pooled.append(out["pooled"].detach().cpu())
            structural.append(out["structural"].detach().cpu())
            nodes.append(out["nodes"].detach().cpu().reshape(out["nodes"].shape[0], -1))
    pooled_t = torch.cat(pooled, dim=0)
    structural_t = torch.cat(structural, dim=0)
    nodes_t = torch.cat(nodes, dim=0)
    summary = {
        "feature_effective_rank": float(effective_rank(pooled_t).item()),
        "structural_effective_rank": float(effective_rank(structural_t).item()),
        "node_effective_rank": float(effective_rank(nodes_t).item()),
        "depth_mean": sum(r["depth"] for r in rows) / len(rows),
        "usage_mean": sum(r["usage"] for r in rows) / len(rows),
        "perplexity_mean": sum(r["perplexity"] for r in rows) / len(rows),
        "node_entropy_mean": sum(r["node_entropy"] for r in rows) / len(rows),
        "stack_alignment_mean": sum(r["stack_alignment"] for r in rows) / len(rows),
        "stack_geodesic_mean": sum(r["stack_geodesic"] for r in rows) / len(rows),
        "gate_mean": sum(r["gate_mean"] for r in rows) / len(rows),
        "probe_accuracy": ridge_probe_accuracy(pooled_t, torch.cat([b["label"].cpu() for b in val_loader], dim=0), cfg.num_classes),
    }
    if len(nodes_t) > 1:
        summary["node_self_overlap"] = float(subspace_overlap(nodes_t[: max(2, min(8, nodes_t.shape[0]))].T, nodes_t[: max(2, min(8, nodes_t.shape[0]))].T).item())
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "diagnostics_rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    prefix = Path(args.output) if args.output else out_dir / "diagnostics.png"
    save_line_plot(prefix.with_name(prefix.stem + "_depth.png"), list(range(len(rows))), [r["depth"] for r in rows], "Depth", "Batch", "Depth")
    save_line_plot(prefix.with_name(prefix.stem + "_usage.png"), list(range(len(rows))), [r["usage"] for r in rows], "Usage", "Batch", "Usage")
    save_line_plot(prefix.with_name(prefix.stem + "_geometry.png"), list(range(len(rows))), [r["stack_alignment"] for r in rows], "Alignment", "Batch", "Alignment")
    save_line_plot(prefix.with_name(prefix.stem + "_gate.png"), list(range(len(rows))), [r["gate_mean"] for r in rows], "Gate Mean", "Batch", "Gate")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
