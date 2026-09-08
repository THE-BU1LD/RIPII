from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import tempfile

# Headless and sandboxed environments may not provide a writable user cache.
# Configure a per-process cache before importing pyplot; otherwise the CLI can
# abort while multiple test processes build the global font cache.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"ripii-matplotlib-{os.getpid()}"),
)
import argparse
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError as exc:
    warnings.warn(f"PyTorch thread limits could not be set: {exc}")

from ripii.models.factory import build_model
from ripii.utils.config import load_config, runtime_profile
from ripii.utils.metrics import (
    effective_rank,
    heldout_ridge_probe_accuracy,
    subspace_overlap,
)
from ripii.utils.training import build_data_splits, load_checkpoint


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
    parser.add_argument("--allow-existing-development", action="store_true")
    args = parser.parse_args()
    cfg = runtime_profile(load_config(args.config))
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    train_loader, _, test_loader = build_data_splits(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    rows = []
    pooled = []
    structural = []
    nodes = []
    labels = []
    train_pooled = []
    train_labels = []
    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward(batch["x"], transform=batch.get("transform"))
            train_pooled.append(out["pooled"].detach().cpu())
            train_labels.append(batch["label"].detach().cpu())
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward(
                batch["x"], x_view=batch["x_view"], transform=batch.get("transform")
            )
            row = {"samples": batch["x"].shape[0]}
            for name, key in {
                "depth": "expected_depth",
                "usage": "vq_usage",
                "perplexity": "vq_coarse_perplexity",
                "node_entropy": "node_entropy",
                "stack_alignment": "stack_alignment",
                "stack_geodesic": "stack_geodesic",
                "gate_mean": "struct_gate_mean",
            }.items():
                row[name] = float(out[key].detach().cpu()) if key in out else None
            rows.append(row)
            pooled.append(out["pooled"].detach().cpu())
            structural.append(out["structural"].detach().cpu())
            if "nodes" in out:
                nodes.append(
                    out["nodes"].detach().cpu().reshape(out["nodes"].shape[0], -1)
                )
            labels.append(batch["label"].detach().cpu())
    pooled_t = torch.cat(pooled, dim=0)
    structural_t = torch.cat(structural, dim=0)
    nodes_t = torch.cat(nodes, dim=0) if nodes else None

    def average(name):
        measured = [row for row in rows if row[name] is not None]
        return (
            sum(row[name] * row["samples"] for row in measured)
            / sum(row["samples"] for row in measured)
            if measured
            else None
        )

    summary = {
        "feature_effective_rank": float(effective_rank(pooled_t).item()),
        "structural_effective_rank": float(effective_rank(structural_t).item()),
        "node_effective_rank": float(effective_rank(nodes_t).item())
        if nodes_t is not None
        else None,
        "depth_mean": average("depth"),
        "usage_mean": average("usage"),
        "perplexity_mean": average("perplexity"),
        "node_entropy_mean": average("node_entropy"),
        "stack_alignment_mean": average("stack_alignment"),
        "stack_geodesic_mean": average("stack_geodesic"),
        "gate_mean": average("gate_mean"),
        "heldout_probe_accuracy": heldout_ridge_probe_accuracy(
            torch.cat(train_pooled, dim=0),
            torch.cat(train_labels, dim=0),
            pooled_t,
            torch.cat(labels, dim=0),
            cfg.num_classes,
        ),
        "probe_protocol": "ridge fit on train features; scored once on test",
        "split_protocol": "70% train / 15% validation / 15% held-out test",
        "profile": cfg.profile,
        "evidence_status": "development_only",
        "quantizer_metrics_applicable": cfg.use_quantizer,
    }
    if not cfg.use_quantizer:
        summary["usage_mean"] = None
        summary["perplexity_mean"] = None
    if nodes_t is not None and len(nodes_t) > 1:
        summary["node_self_overlap"] = float(
            subspace_overlap(
                nodes_t[: max(2, min(8, nodes_t.shape[0]))].T,
                nodes_t[: max(2, min(8, nodes_t.shape[0]))].T,
            ).item()
        )
    checkpoint_path = Path(args.checkpoint)
    prefix = (
        Path(args.output) if args.output else checkpoint_path.parent / "diagnostics.png"
    )
    out_dir = prefix.parent
    outputs = [
        prefix.with_name(prefix.stem + "_features.png"),
        out_dir / "diagnostics.json",
        out_dir / "diagnostics_rows.json",
        prefix.with_name(prefix.stem + "_depth.png"),
        prefix.with_name(prefix.stem + "_usage.png"),
        prefix.with_name(prefix.stem + "_geometry.png"),
        prefix.with_name(prefix.stem + "_gate.png"),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.allow_existing_development:
        raise SystemExit(f"refusing to overwrite diagnostics output: {existing[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    with open(out_dir / "diagnostics_rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, allow_nan=False)
    for suffix, key, title in (
        ("depth", "depth", "Depth"),
        ("usage", "usage", "Usage"),
        ("geometry", "stack_alignment", "Alignment"),
        ("gate", "gate_mean", "Gate mean"),
    ):
        if all(row[key] is not None for row in rows):
            save_line_plot(
                prefix.with_name(prefix.stem + f"_{suffix}.png"),
                list(range(len(rows))),
                [row[key] for row in rows],
                title,
                "Batch",
                title,
            )
    save_line_plot(
        prefix.with_name(prefix.stem + "_features.png"),
        list(range(len(pooled_t))),
        pooled_t.norm(dim=-1).tolist(),
        "Held-out feature norms",
        "Example",
        "Norm",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
