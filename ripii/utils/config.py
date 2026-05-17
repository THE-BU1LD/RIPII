
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import os
import yaml


@dataclass
class LossWeights:
    rec: float = 1.0
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
    def from_mapping(cls, mapping: dict[str, Any]) -> "Config":
        mapping = dict(mapping or {})
        loss = mapping.pop("loss_weights", {})
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in mapping.items() if k in allowed}
        cfg = cls(**filtered)
        cfg.loss_weights = LossWeights(**{k: v for k, v in dict(loss).items() if k in LossWeights.__dataclass_fields__})
        return cfg

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["loss_weights"] = asdict(self.loss_weights)
        return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        data.update(loaded)
    if overrides:
        data.update(overrides)
    out_dir = os.environ.get("RIPII_OUTPUT_DIR")
    if out_dir:
        data["output_dir"] = out_dir
    return Config.from_mapping(data)


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.as_dict(), f, sort_keys=False)


def runtime_profile(cfg: Config) -> Config:
    small = cfg.input_dim <= 32 and cfg.dataset_size <= 64 and cfg.steps <= 12
    if not small:
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
