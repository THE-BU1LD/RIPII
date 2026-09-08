from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError as exc:
    warnings.warn(f"PyTorch thread limits could not be set: {exc}")

from ripii.models.factory import build_model
from ripii.utils.config import load_config, runtime_profile
from ripii.utils.metrics import (
    cosine_similarity,
    effective_rank,
    heldout_ridge_probe_accuracy,
    mse,
)
from ripii.utils.training import build_data_splits, load_checkpoint


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
    totals = {}
    sample_count = 0
    test_pooled_features = []
    test_structural_features = []
    test_labels = []
    train_pooled_features = []
    train_structural_features = []
    train_labels = []
    consistency = []
    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward(batch["x"], transform=batch.get("transform"))
            train_pooled_features.append(out["pooled"].detach().cpu())
            train_structural_features.append(out["structural"].detach().cpu())
            train_labels.append(batch["label"].detach().cpu())
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses = model.losses(batch, vars(cfg.loss_weights), warmup=False)
            out = model.forward(
                batch["x"], x_view=batch["x_view"], transform=batch.get("transform")
            )
            test_pooled_features.append(out["pooled"].detach().cpu())
            test_structural_features.append(out["structural"].detach().cpu())
            test_labels.append(batch["label"].detach().cpu())
            batch_size = batch["x"].shape[0]
            sample_count += batch_size
            consistency.append(
                {
                    "samples": batch_size,
                    "latent_cosine": float(
                        cosine_similarity(
                            out["pooled"],
                            out["view_pooled"]
                            if "view_pooled" in out
                            else out["pooled"],
                        )
                        .detach()
                        .cpu()
                    ),
                    "structural_mse": float(
                        mse(
                            out["structural"],
                            out["view_structural"]
                            if "view_structural" in out
                            else out["structural"],
                        )
                        .detach()
                        .cpu()
                    ),
                    "quantized_cosine": float(
                        cosine_similarity(
                            out["quantized"],
                            out["view_quantized"]
                            if "view_quantized" in out
                            else out["quantized"],
                        )
                        .detach()
                        .cpu()
                    ),
                }
            )
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    totals[k] = totals.get(k, 0.0) + batch_size * float(
                        v.detach().cpu()
                    )
                else:
                    totals[k] = totals.get(k, 0.0) + batch_size * float(v)
    n = sample_count
    if n == 0:
        raise RuntimeError("test loader is empty")
    summary = {k: v / n for k, v in totals.items()}
    train_pooled = torch.cat(train_pooled_features, dim=0)
    train_structural = torch.cat(train_structural_features, dim=0)
    train_labs = torch.cat(train_labels, dim=0)
    test_pooled = torch.cat(test_pooled_features, dim=0)
    test_structural = torch.cat(test_structural_features, dim=0)
    test_labs = torch.cat(test_labels, dim=0)
    summary["heldout_probe_accuracy"] = heldout_ridge_probe_accuracy(
        train_pooled, train_labs, test_pooled, test_labs, cfg.num_classes
    )
    summary["heldout_structural_probe_accuracy"] = heldout_ridge_probe_accuracy(
        train_structural,
        train_labs,
        test_structural,
        test_labs,
        cfg.num_classes,
    )
    summary["feature_effective_rank"] = float(effective_rank(test_pooled).item())
    summary["structural_effective_rank"] = float(effective_rank(test_structural).item())
    if consistency:
        summary["latent_cosine_mean"] = (
            sum(c["samples"] * c["latent_cosine"] for c in consistency) / sample_count
        )
        summary["structural_mse_mean"] = (
            sum(c["samples"] * c["structural_mse"] for c in consistency) / sample_count
        )
        summary["quantized_cosine_mean"] = (
            sum(c["samples"] * c["quantized_cosine"] for c in consistency)
            / sample_count
        )
    summary["profile"] = cfg.profile
    summary["model_variant"] = cfg.model_variant
    summary["evidence_status"] = "development_only"
    summary["split_protocol"] = "70% train / 15% validation / 15% held-out test"
    summary["probe_protocol"] = "ridge fit on train features; scored once on test"
    summary["active_mechanisms"] = {
        "projective": cfg.use_projective,
        "graph": cfg.use_graph,
        "quantizer": cfg.use_quantizer,
        "action": cfg.use_action,
    }
    summary["quantizer_metrics_applicable"] = cfg.use_quantizer
    if cfg.use_quantizer:
        summary["coarse_effective_code_fraction"] = (
            summary["perplexity_coarse"] / cfg.codebook_size
        )
        summary["fine_effective_code_fraction"] = (
            summary["perplexity_fine"] / cfg.fine_codebook_size
        )
    else:
        # The model carries finite neutral tensors through its common forward
        # schema, but they are not measurements when quantization is bypassed.
        for key in (
            "perplexity_coarse",
            "perplexity_fine",
            "usage",
            "quantized_cosine_mean",
        ):
            summary[key] = None
    summary["total_params"] = sum(parameter.numel() for parameter in model.parameters())
    summary["trainable_params"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    checkpoint_path = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint_path.parent / "eval.json"
    if output.exists() and not args.allow_existing_development:
        raise SystemExit(f"refusing to overwrite {output}; select a fresh --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(
        output, "x" if not args.allow_existing_development else "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
