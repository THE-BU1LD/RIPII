from __future__ import annotations

from typing import Protocol

from .baselines import PlainAutoencoder
from .ripii import RIPIIModel


class ModelConfig(Protocol):
    model_variant: str
    input_dim: int
    latent_dim: int
    hidden_dim: int
    node_dim: int
    num_nodes: int
    num_levels: int
    num_projectors: int
    codebook_size: int
    codebook_dim: int
    graph_steps: int
    transform_dim: int
    fine_codebook_size: int
    depth_target: float
    graph_topk: int
    use_projective: bool
    use_graph: bool
    use_quantizer: bool
    use_action: bool
    use_spectral_loss: bool
    use_equivariance_loss: bool
    use_identity_loss: bool
    use_depth_loss: bool


def build_model(cfg: ModelConfig) -> RIPIIModel | PlainAutoencoder:
    if cfg.model_variant == "plain_ae":
        return PlainAutoencoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
        )
    if cfg.model_variant != "ripii":
        raise ValueError(f"unknown model_variant: {cfg.model_variant}")
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
