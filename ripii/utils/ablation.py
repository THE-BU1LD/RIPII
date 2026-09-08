from __future__ import annotations

from dataclasses import replace

from .config import Config


def apply_mode(cfg: Config, mode: str) -> Config:
    mode = mode or "base"
    if mode == "base":
        return cfg
    if mode == "plain_ae":
        return replace(
            cfg,
            model_variant="plain_ae",
            use_projective=False,
            num_levels=0,
            use_graph=False,
            graph_steps=0,
            graph_topk=0,
            use_quantizer=False,
            codebook_size=1,
            fine_codebook_size=1,
            use_action=False,
            use_spectral_loss=False,
            use_equivariance_loss=False,
            use_identity_loss=False,
            use_depth_loss=False,
            loss_weights=replace(
                cfg.loss_weights,
                equiv=0.0,
                inv=0.0,
                scale=0.0,
                proj=0.0,
                spectral=0.0,
                geom=0.0,
                vq=0.0,
                node=0.0,
                moment=0.0,
                identity=0.0,
                depth=0.0,
            ),
        )
    if mode == "no_renorm":
        return replace(
            cfg,
            use_projective=False,
            num_levels=1,
            num_projectors=1,
            use_spectral_loss=False,
            use_depth_loss=False,
            loss_weights=replace(
                cfg.loss_weights, scale=0.0, proj=0.0, spectral=0.0, depth=0.0, geom=0.0
            ),
        )
    if mode == "no_graph":
        return replace(
            cfg,
            use_graph=False,
            graph_steps=0,
            graph_topk=0,
            loss_weights=replace(cfg.loss_weights, node=0.0),
        )
    if mode == "no_vq":
        return replace(
            cfg,
            use_quantizer=False,
            codebook_size=1,
            fine_codebook_size=1,
            loss_weights=replace(cfg.loss_weights, vq=0.0),
        )
    if mode == "no_action":
        return replace(
            cfg,
            use_action=False,
            loss_weights=replace(cfg.loss_weights, equiv=0.0, identity=0.0),
        )
    if mode == "no_geom":
        return replace(
            cfg,
            use_spectral_loss=False,
            loss_weights=replace(cfg.loss_weights, proj=0.0, spectral=0.0, geom=0.0),
        )
    if mode == "no_scale":
        return replace(cfg, loss_weights=replace(cfg.loss_weights, scale=0.0))
    if mode == "no_identity":
        return replace(cfg, loss_weights=replace(cfg.loss_weights, identity=0.0))
    if mode == "no_equiv":
        return replace(cfg, loss_weights=replace(cfg.loss_weights, equiv=0.0))
    if mode == "no_spectral":
        return replace(
            cfg,
            use_spectral_loss=False,
            loss_weights=replace(cfg.loss_weights, spectral=0.0),
        )
    if mode == "no_depth":
        return replace(
            cfg, use_depth_loss=False, loss_weights=replace(cfg.loss_weights, depth=0.0)
        )
    if mode == "no_moment":
        return replace(cfg, loss_weights=replace(cfg.loss_weights, moment=0.0))
    if mode == "no_structured":
        return replace(
            cfg,
            use_projective=False,
            num_levels=0,
            use_graph=False,
            graph_steps=0,
            graph_topk=0,
            use_quantizer=False,
            codebook_size=1,
            fine_codebook_size=1,
            use_action=False,
            use_spectral_loss=False,
            use_equivariance_loss=False,
            use_identity_loss=False,
            use_depth_loss=False,
            loss_weights=replace(
                cfg.loss_weights,
                equiv=0.0,
                scale=0.0,
                proj=0.0,
                spectral=0.0,
                geom=0.0,
                vq=0.0,
                node=0.0,
                identity=0.0,
                depth=0.0,
            ),
        )
    raise ValueError(f"unknown mode: {mode}")
