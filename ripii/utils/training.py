from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..data.synthetic import SyntheticStructuralDataset


def collate(samples):
    return {
        "x": torch.stack([s.x for s in samples]),
        "x_view": torch.stack([s.x_view for s in samples]),
        "transform": torch.stack([s.transform for s in samples]),
        "label": torch.stack([s.label for s in samples]),
    }


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticStructuralDataset(
        size=cfg.dataset_size,
        input_dim=cfg.input_dim,
        noise_std=cfg.noise_std,
        transform_scale=cfg.transform_scale,
        transform_shift=cfg.transform_shift,
        transform_dim=cfg.transform_dim,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )
    val_size = max(1, int(0.15 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def move_batch(batch, device: str):
    return {k: v.to(device) for k, v in batch.items()}


def grad_norms(model) -> dict[str, float]:
    groups = {
        "encoder": model.encoder,
        "projective": model.renorm,
        "graph": model.graph,
        "quantizer": model.quantizer,
        "decoder": model.decoder,
        "action": model.action,
    }
    values = {}
    for name, module in groups.items():
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().cpu())
        values[f"grad_{name}"] = total ** 0.5
    return values


def run_epoch(model, loader, optimizer, cfg, device: str, train: bool, step_offset: int = 0, scaler=None):
    model.train(train)
    totals = {}
    iterator = tqdm(loader, desc="train" if train else "val", leave=False, disable=not sys.stderr.isatty())
    amp = device.startswith("cuda") and scaler is not None and getattr(cfg, "amp", False)
    for step, batch in enumerate(iterator):
        batch = move_batch(batch, device)
        warmup = train and (step + step_offset) < cfg.warmup_steps
        with torch.set_grad_enabled(train):
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    losses = model.losses(batch, asdict(cfg.loss_weights), warmup=warmup)
                    loss = losses["total"]
            else:
                losses = model.losses(batch, asdict(cfg.loss_weights), warmup=warmup)
                loss = losses["total"]
            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
            else:
                totals[k] = totals.get(k, 0.0) + float(v)
        if train:
            for k, v in grad_norms(model).items():
                totals[k] = totals.get(k, 0.0) + float(v)
        iterator.set_postfix({"loss": float(loss.detach().cpu())})
    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


def save_checkpoint(path: str | Path, model, optimizer, cfg, step: int, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": asdict(cfg), "step": step, "metrics": metrics}, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
