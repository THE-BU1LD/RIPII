from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

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


def build_data_splits(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
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
    test_size = max(1, int(0.15 * len(ds)))
    train_size = len(ds) - val_size - test_size
    if train_size < 1:
        raise ValueError(
            "dataset_size must leave non-empty train, validation, and test splits"
        )
    train_ds, val_ds, test_ds = random_split(
        ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg.seed + 31),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg.seed + 37),
    )
    return train_loader, val_loader, test_loader


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    train_loader, val_loader, _ = build_data_splits(cfg)
    return train_loader, val_loader


def move_batch(batch, device: str):
    return {k: v.to(device) for k, v in batch.items()}


def grad_norms(model) -> dict[str, float]:
    groups = {
        name: getattr(model, attribute)
        for name, attribute in (
            ("encoder", "encoder"),
            ("projective", "renorm"),
            ("graph", "graph"),
            ("quantizer", "quantizer"),
            ("decoder", "decoder"),
            ("action", "action"),
        )
        if hasattr(model, attribute)
    }
    values = {}
    for name, module in groups.items():
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().cpu())
        values[f"grad_{name}"] = total**0.5
    return values


def run_epoch(
    model,
    loader,
    optimizer,
    cfg,
    device: str,
    train: bool,
    step_offset: int = 0,
    scaler=None,
):
    model.train(train)
    totals = {}
    sample_count = 0
    iterator = tqdm(
        loader,
        desc="train" if train else "val",
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    amp = (
        device.startswith("cuda") and scaler is not None and getattr(cfg, "amp", False)
    )
    for step, batch in enumerate(iterator):
        batch = move_batch(batch, device)
        warmup = train and (step + step_offset) < cfg.warmup_steps
        with torch.set_grad_enabled(train):
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    losses = model.losses(
                        batch, asdict(cfg.loss_weights), warmup=warmup
                    )
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
        batch_size = batch["x"].shape[0]
        sample_count += batch_size
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                totals[k] = totals.get(k, 0.0) + batch_size * float(v.detach().cpu())
            else:
                totals[k] = totals.get(k, 0.0) + batch_size * float(v)
        if train:
            for k, v in grad_norms(model).items():
                totals[k] = totals.get(k, 0.0) + batch_size * float(v)
        iterator.set_postfix({"loss": float(loss.detach().cpu())})
    n = max(1, sample_count)
    return {k: v / n for k, v in totals.items()}


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    cfg,
    step: int,
    metrics: dict,
    training_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model": getattr(model, "_orig_mod", model).state_dict(),
            "checkpoint_version": 2,
            "training_state": training_state,
            "optimizer": optimizer.state_dict(),
            "cfg": asdict(cfg),
            "step": step,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu"):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint must be a regular non-symlink file: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model"), dict):
        raise ValueError("checkpoint has an invalid schema")
    getattr(model, "_orig_mod", model).load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def load_compatible_initial_state(path: str | Path, model) -> dict[str, int | str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"initial state must be a regular non-symlink file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("initial state has an invalid schema")
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    if not compatible:
        raise ValueError("initial state contains no compatible parameters")
    model.load_state_dict(compatible, strict=False)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "compatible_tensor_count": len(compatible),
        "source_tensor_count": len(state),
        "target_tensor_count": len(current),
    }
