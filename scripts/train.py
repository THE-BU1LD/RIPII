from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import platform
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
from ripii.utils.ablation import apply_mode
from ripii.utils.config import (
    load_config,
    runtime_profile,
    save_config,
    validate_config,
)
from ripii.utils.seed import seed_everything
from ripii.utils.training import (
    build_dataloaders,
    collate,
    load_checkpoint,
    load_compatible_initial_state,
    run_epoch,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--initial-state", type=str, default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-existing-development",
        action="store_true",
        help="permit writes into a non-empty output directory; never use for retained evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.initial_state:
        raise SystemExit("--resume and --initial-state are mutually exclusive")
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
    validate_config(cfg)
    seed_everything(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    initialization = None
    if args.initial_state:
        initialization = load_compatible_initial_state(args.initial_state, model)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    )
    start_step = 0
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, opt, map_location=str(device))
        if ckpt.get("checkpoint_version") != 2 or not ckpt.get("training_state"):
            raise SystemExit(
                "legacy checkpoint cannot resume exactly; use --initial-state for a new run"
            )
        for key, value in cfg.as_dict().items():
            if (
                key not in {"steps", "output_dir", "device", "compile_model"}
                and ckpt["cfg"].get(key) != value
            ):
                raise SystemExit(
                    f"resume configuration differs at {key}; use the saved config"
                )
        start_step = int(ckpt["step"])
        if start_step >= cfg.steps:
            raise SystemExit(
                f"checkpoint already completed {start_step} steps; increase --steps"
            )
    seed_everything(cfg.seed + 1_000_003)
    if ckpt is not None:
        torch.set_rng_state(ckpt["training_state"]["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ckpt["training_state"]["cuda_rng"])
        scaler.load_state_dict(ckpt["training_state"]["scaler"])
    out_dir = Path(
        args.output_dir or os.environ.get("RIPII_OUTPUT_DIR") or cfg.output_dir
    )
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and not args.resume
        and not args.allow_existing_development
    ):
        raise SystemExit(
            f"refusing to overwrite non-empty run directory {out_dir}; choose a fresh --output-dir"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    save_config(cfg, out_dir / "config.yaml")
    metadata = {
        "evidence_status": "development_only",
        "profile": cfg.profile,
        "seed": cfg.seed,
        "mode": args.mode,
        "model_variant": cfg.model_variant,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "initialization": initialization,
        "training_rng_seed": cfg.seed + 1_000_003,
    }
    if not args.resume or not (out_dir / "run_metadata.json").exists():
        metadata["resumed_from"] = str(Path(args.resume).name) if args.resume else None
        metadata_mode = "w" if args.allow_existing_development else "x"
        with open(out_dir / "run_metadata.json", metadata_mode, encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, allow_nan=False)
    history = []
    best_val = ckpt["training_state"]["best_val"] if ckpt is not None else None
    if args.resume and not (out_dir / "best.pt").exists():
        best_source = Path(args.resume).parent / "best.pt"
        if best_source.is_file():
            import shutil

            shutil.copy2(best_source, out_dir / "best.pt")
        else:
            best_val = None
    should_print = sys.stdout.isatty() or os.environ.get("RIPII_FORCE_STDOUT") == "1"

    def continuation_state():
        return {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "scaler": scaler.state_dict(),
            "best_val": best_val,
        }

    # A step-addressable sampler makes minibatch order independent of restart.
    train_size = len(train_loader.dataset)
    batches_per_epoch = len(train_loader)
    for step in range(start_step, cfg.steps):
        epoch, batch_index = divmod(step, batches_per_epoch)
        order = torch.randperm(
            train_size,
            generator=torch.Generator().manual_seed(cfg.seed + 2_000_003 + epoch),
        )
        indices = order[
            batch_index * cfg.batch_size : (batch_index + 1) * cfg.batch_size
        ]
        batch = collate([train_loader.dataset[int(index)] for index in indices])
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
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    opt,
                    cfg,
                    str(device),
                    train=False,
                    step_offset=step,
                    scaler=scaler,
                )
            record = {"step": step + 1, "train_total": float(loss.detach().cpu())}
            record.update(
                {
                    f"train_{k}": float(v.detach().cpu())
                    for k, v in losses.items()
                    if isinstance(v, torch.Tensor) and k != "total"
                }
            )
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(record)
            if should_print:
                print(json.dumps(record))
            with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            val_total = val_metrics.get("total", float("inf"))
            if best_val is None or val_total < best_val:
                best_val = val_total
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    opt,
                    cfg,
                    step + 1,
                    record,
                    continuation_state(),
                )
            save_checkpoint(
                out_dir / "latest.pt",
                model,
                opt,
                cfg,
                step + 1,
                record,
                continuation_state(),
            )
    save_checkpoint(
        out_dir / "final.pt",
        model,
        opt,
        cfg,
        cfg.steps,
        history[-1],
        continuation_state(),
    )


if __name__ == "__main__":
    main()
