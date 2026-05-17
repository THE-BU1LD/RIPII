from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import os
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
from ripii.utils.ablation import apply_mode
from ripii.utils.config import load_config, runtime_profile, save_config
from ripii.utils.seed import seed_everything
from ripii.utils.training import build_dataloaders, load_checkpoint, run_epoch, save_checkpoint


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = runtime_profile(load_config(args.config))
    if args.steps is not None:
        cfg.steps = int(args.steps)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.amp:
        cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    cfg = apply_mode(cfg, args.mode)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, opt, map_location=str(device))
        start_step = int(ckpt.get("step", 0))
    out_dir = Path(args.output_dir or os.environ.get("RIPII_OUTPUT_DIR") or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    save_config(cfg, out_dir / "config.yaml")
    history = []
    best_val = None
    should_print = sys.stdout.isatty() or os.environ.get("RIPII_FORCE_STDOUT") == "1"
    train_iter = iter(train_loader)
    for step in range(start_step, cfg.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        warmup = step < cfg.warmup_steps
        model.train(True)
        if cfg.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                losses = model.losses(batch, vars(cfg.loss_weights), warmup=warmup)
                loss = losses["total"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            losses = model.losses(batch, vars(cfg.loss_weights), warmup=warmup)
            loss = losses["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            model.eval()
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, opt, cfg, str(device), train=False, step_offset=step, scaler=scaler)
            record = {"step": step, "train_total": float(loss.detach().cpu())}
            record.update({f"train_{k}": float(v.detach().cpu()) for k, v in losses.items() if isinstance(v, torch.Tensor) and k != "total"})
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(record)
            if should_print:
                print(json.dumps(record))
            save_checkpoint(out_dir / "latest.pt", model, opt, cfg, step, record)
            with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            val_total = val_metrics.get("total", float("inf"))
            if best_val is None or val_total < best_val:
                best_val = val_total
                save_checkpoint(out_dir / "best.pt", model, opt, cfg, step, record)
    if not history:
        history.append({"step": cfg.steps, "train_total": float(loss.detach().cpu())})
    save_checkpoint(out_dir / "final.pt", model, opt, cfg, cfg.steps, history[-1])


if __name__ == "__main__":
    main()
