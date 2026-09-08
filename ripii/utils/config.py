from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LossWeights:
    recon: float = 1.0
    equiv: float = 0.8
    inv: float = 0.8
    scale: float = 0.3
    proj: float = 0.2
    spectral: float = 0.15
    geom: float = 0.05
    vq: float = 1.0
    node: float = 0.1
    moment: float = 0.05
    identity: float = 0.05
    kl: float = 0.01
    depth: float = 0.05


@dataclass
class Config:
    model_variant: str = "ripii"
    profile: str = "research"
    seed: int = 7
    device: str = "cpu"
    output_dir: str = "runs/ripii"
    batch_size: int = 64
    val_batch_size: int = 128
    steps: int = 600
    warmup_steps: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-2
    input_dim: int = 128
    latent_dim: int = 96
    hidden_dim: int = 192
    node_dim: int = 24
    num_nodes: int = 4
    num_levels: int = 3
    num_projectors: int = 3
    codebook_size: int = 64
    fine_codebook_size: int = 64
    codebook_dim: int = 64
    graph_steps: int = 3
    graph_topk: int = 2
    dataset_size: int = 4096
    noise_std: float = 0.07
    transform_scale: float = 0.25
    transform_shift: int = 3
    transform_dim: int = 4
    num_classes: int = 8
    eval_every: int = 20
    log_every: int = 20
    compile_model: bool = False
    amp: bool = True
    depth_target: float = 0.55
    use_projective: bool = True
    use_graph: bool = True
    use_quantizer: bool = True
    use_action: bool = True
    use_spectral_loss: bool = True
    use_equivariance_loss: bool = True
    use_identity_loss: bool = True
    use_depth_loss: bool = True
    loss_weights: LossWeights = field(default_factory=LossWeights)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Config":
        if not isinstance(mapping, Mapping):
            raise ValueError("configuration must be a mapping")
        mapping = dict(mapping)
        loss = mapping.pop("loss_weights", {})
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
        if not isinstance(loss, Mapping):
            raise ValueError("loss_weights must be a mapping")
        loss = dict(loss)
        # Historical configs used ``rec`` while the model objective is named
        # ``recon``.  Accept the old spelling explicitly so retained runs remain
        # readable, but never allow both spellings to disagree.
        if "rec" in loss:
            if "recon" in loss:
                raise ValueError("loss_weights cannot contain both rec and recon")
            loss["recon"] = loss.pop("rec")
        allowed_loss = {item.name for item in fields(LossWeights)}
        unknown_loss = sorted(set(loss) - allowed_loss)
        if unknown_loss:
            raise ValueError(f"unknown loss-weight keys: {', '.join(unknown_loss)}")
        cfg = cls(**mapping)
        cfg.loss_weights = LossWeights(**loss)
        validate_config(cfg)
        return cfg

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["loss_weights"] = asdict(self.loss_weights)
        return data


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> Config:
    data: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("configuration document must contain a mapping")
        data.update(loaded)
    if overrides:
        data.update(overrides)
    out_dir = os.environ.get("RIPII_OUTPUT_DIR")
    if out_dir:
        data["output_dir"] = out_dir
    return Config.from_mapping(data)


def validate_config(cfg: Config) -> None:
    if cfg.model_variant not in {"ripii", "plain_ae"}:
        raise ValueError("model_variant must be ripii or plain_ae")
    if cfg.profile not in {"research", "plumbing_smoke", "mechanism_smoke"}:
        raise ValueError("profile must be research, plumbing_smoke, or mechanism_smoke")

    positive_ints = (
        "batch_size",
        "val_batch_size",
        "steps",
        "input_dim",
        "latent_dim",
        "hidden_dim",
        "node_dim",
        "num_nodes",
        "num_projectors",
        "codebook_size",
        "fine_codebook_size",
        "codebook_dim",
        "dataset_size",
        "transform_dim",
        "num_classes",
        "eval_every",
        "log_every",
    )
    for name in positive_ints:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "warmup_steps",
        "num_levels",
        "graph_steps",
        "graph_topk",
        "transform_shift",
    ):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    if cfg.transform_dim != 4:
        raise ValueError(
            "transform_dim must be exactly 4 for the implemented transform"
        )
    if cfg.input_dim < max(8, cfg.num_classes + 4):
        raise ValueError("input_dim is too small for the configured semantic basis")
    if cfg.dataset_size < 3:
        raise ValueError(
            "dataset_size must provide train, validation, and test examples"
        )
    if cfg.use_graph and cfg.graph_steps > 0 and not 0 < cfg.graph_topk < cfg.num_nodes:
        raise ValueError(
            "graph_topk must be between 1 and num_nodes - 1 when graph refinement is active"
        )

    finite_names = (
        "lr",
        "weight_decay",
        "noise_std",
        "transform_scale",
        "depth_target",
    )
    for name in finite_names:
        value = getattr(cfg, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be finite")
    if (
        cfg.lr <= 0
        or cfg.weight_decay < 0
        or cfg.noise_std < 0
        or cfg.transform_scale < 0
    ):
        raise ValueError(
            "lr must be positive and regularization/noise/transform scales non-negative"
        )
    if not 0.0 <= cfg.depth_target <= 1.0:
        raise ValueError("depth_target must be in [0, 1]")
    for name, value in asdict(cfg.loss_weights).items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"loss weight {name} must be finite and non-negative")


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.as_dict(), f, sort_keys=False)


def runtime_profile(cfg: Config) -> Config:
    if cfg.profile != "plumbing_smoke":
        return cfg
    return replace(
        cfg,
        batch_size=max(1, min(cfg.batch_size, 4)),
        val_batch_size=max(1, min(cfg.val_batch_size, 4)),
        latent_dim=max(4, min(cfg.latent_dim, 16)),
        hidden_dim=max(8, min(cfg.hidden_dim, 32)),
        node_dim=max(4, min(cfg.node_dim, 4)),
        num_nodes=max(2, min(cfg.num_nodes, 3)),
        num_levels=0,
        num_projectors=1,
        codebook_size=max(4, min(cfg.codebook_size, 4)),
        fine_codebook_size=max(4, min(cfg.fine_codebook_size, 4)),
        codebook_dim=max(4, min(cfg.codebook_dim, 8)),
        graph_steps=0,
        graph_topk=0,
        use_projective=False,
        use_graph=False,
        use_quantizer=False,
        use_action=False,
    )
