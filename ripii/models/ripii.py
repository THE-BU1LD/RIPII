from __future__ import annotations

import torch
from torch import nn

from ..utils.loss_balancer import AdaptiveLossBalancer
from ..utils.metrics import (
    cosine_distance,
    covariance_penalty,
    effective_rank,
    mse,
    sanitize,
    spectral_distance,
    variance_penalty,
)
from .decoder import ResidualDecoder
from .encoder import ResidualEncoder
from .graph import LatentGraphModule
from .latent_action import LatentActionModule
from .layers import kl_divergence
from .projective import ProjectiveRenormStack
from .quantizer import HierarchicalVectorQuantizer


class RIPIIModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        node_dim: int,
        num_nodes: int,
        num_levels: int,
        num_projectors: int,
        codebook_size: int,
        codebook_dim: int,
        graph_steps: int,
        transform_dim: int = 4,
        fine_codebook_size: int | None = None,
        depth_target: float = 0.55,
        graph_topk: int = 2,
        use_projective: bool = True,
        use_graph: bool = True,
        use_quantizer: bool = True,
        use_action: bool = True,
        use_spectral_loss: bool = True,
        use_equivariance_loss: bool = True,
        use_identity_loss: bool = True,
        use_depth_loss: bool = True,
    ) -> None:
        super().__init__()

        fine_codebook_size = fine_codebook_size or codebook_size

        self.encoder = ResidualEncoder(input_dim, hidden_dim, latent_dim)
        self.use_projective = bool(use_projective)
        self.use_graph = bool(use_graph)
        self.use_quantizer = bool(use_quantizer)
        self.use_action = bool(use_action)
        self.use_spectral_loss = bool(use_spectral_loss)
        self.use_equivariance_loss = bool(use_equivariance_loss)
        self.use_identity_loss = bool(use_identity_loss)
        self.use_depth_loss = bool(use_depth_loss)

        self.renorm = ProjectiveRenormStack(
            latent_dim,
            num_levels if use_projective else 0,
            num_projectors,
        )

        self.to_nodes = nn.Linear(latent_dim, num_nodes * node_dim)
        self.node_post = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )
        self.graph = LatentGraphModule(
            node_dim,
            graph_steps if use_graph else 0,
            topk=graph_topk,
        )
        self.action = LatentActionModule(latent_dim, transform_dim)

        fusion_in = latent_dim * 3 + node_dim

        self.context_proj = nn.Sequential(
            nn.Linear(latent_dim, codebook_dim),
            nn.GELU(),
            nn.LayerNorm(codebook_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, codebook_dim),
            nn.GELU(),
            nn.LayerNorm(codebook_dim),
        )

        self.struct_gate = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_dim),
            nn.Sigmoid(),
        )

        self.quantizer = HierarchicalVectorQuantizer(
            codebook_size,
            fine_codebook_size,
            codebook_dim,
        )

        self.decoder = ResidualDecoder(
            codebook_dim + node_dim + latent_dim,
            hidden_dim,
            input_dim,
        )

        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.latent_dim = int(latent_dim)
        self.num_levels = max(0, num_levels if use_projective else 0)

        self.depth_target = float(depth_target)
        self.register_buffer(
            "_depth_target",
            torch.tensor(float(depth_target), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_num_nodes_tensor",
            torch.tensor(float(max(1, self.num_nodes)), dtype=torch.float32),
            persistent=False,
        )

        self.loss_balancer = AdaptiveLossBalancer(
            [
                "recon",
                "equiv",
                "inv",
                "scale",
                "proj",
                "spectral",
                "geom",
                "vq",
                "node",
                "moment",
                "identity",
                "kl",
                "depth",
            ]
        )

        # Do not expose parameters to the optimizer when their modules are
        # bypassed.  Besides making parameter counts honest, this prevents
        # unused ablation parameters from acquiring optimizer state.
        for enabled, module in (
            (self.use_graph, self.graph),
            (self.use_quantizer, self.quantizer),
            (self.use_action, self.action),
        ):
            if not enabled:
                module.requires_grad_(False)

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    def _stack_stat(
        self,
        stats: dict[str, torch.Tensor],
        key: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        value = stats.get(key)
        if value is None:
            return self._zero(ref)
        return sanitize(value)

    def split_nodes(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        nodes = self.to_nodes(z).view(b, self.num_nodes, self.node_dim)
        return self.node_post(nodes)

    def encode_hierarchy(
        self,
        x: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        z0, mu, logvar = self.encoder(x)
        stages, stack_stats = self.renorm(z0)
        return stages, stack_stats, mu, logvar, stages[-1]

    def _structural_context(
        self,
        z: torch.Tensor,
        transform: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_action and transform is not None:
            return self.action(z, transform)
        return z

    def _forward_core(
        self,
        x: torch.Tensor,
        transform: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stages, stack_stats, mu, logvar, z = self.encode_hierarchy(x)
        nodes = self.split_nodes(z)
        pooled, graph_stats = self.graph(nodes)

        context = self._structural_context(stages[0], transform)
        structural_input = torch.cat([z, pooled, mu, context], dim=-1)

        base_structural = self.fusion(structural_input)
        context_structural = self.context_proj(context)
        gate = self.struct_gate(structural_input)

        structural = gate * base_structural + (1.0 - gate) * context_structural
        structural = sanitize(structural)

        if self.use_quantizer:
            quantized, vq_stats = self.quantizer(structural)
        else:
            quantized = structural
            vq_stats = {
                "vq_commit": self._zero(x),
                "vq_code": self._zero(x),
                "vq_coarse_usage": torch.ones((), device=x.device, dtype=x.dtype),
                "vq_fine_usage": torch.ones((), device=x.device, dtype=x.dtype),
                "vq_coarse_entropy": self._zero(x),
                "vq_fine_entropy": self._zero(x),
                "vq_coarse_perplexity": torch.ones((), device=x.device, dtype=x.dtype),
                "vq_fine_perplexity": torch.ones((), device=x.device, dtype=x.dtype),
                "vq_usage": torch.ones((), device=x.device, dtype=x.dtype),
                "vq_residual_energy": self._zero(x),
            }

        recon = self.decoder(torch.cat([quantized, pooled, z], dim=-1))

        if self.num_levels > 0:
            active_depths = [
                self._stack_stat(stack_stats, f"renorm_{i}_active_depth", x)
                for i in range(self.num_levels)
            ]
            expected_depth = torch.stack(active_depths).mean()
        else:
            expected_depth = self._zero(x)

        out = {
            "recon": sanitize(recon),
            "latent": sanitize(stages[0]),
            "stages": [sanitize(stage) for stage in stages],
            "nodes": sanitize(nodes),
            "pooled": sanitize(pooled),
            "structural": sanitize(structural),
            "quantized": sanitize(quantized),
            "mu": sanitize(mu),
            "logvar": sanitize(logvar),
            "expected_depth": sanitize(expected_depth),
            "effective_rank": effective_rank(structural),
            "transform_context": sanitize(context),
            "struct_gate_mean": sanitize(gate.mean()),
        }

        out.update(stack_stats)
        out.update(graph_stats)
        out.update(vq_stats)

        if self.use_action and transform is not None:
            out["action_latent"] = sanitize(context)
            out["identity_latent"] = sanitize(
                self.action(stages[0], torch.zeros_like(transform))
            )
        elif transform is not None:
            out["action_latent"] = sanitize(stages[0])
            out["identity_latent"] = sanitize(stages[0])

        return out

    def forward(
        self,
        x: torch.Tensor,
        x_view: torch.Tensor | None = None,
        transform: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self._forward_core(x, transform=transform)

        if x_view is not None:
            view_out = self._forward_core(x_view, transform=transform)
            out["view_latent"] = view_out["latent"]
            out["view_pooled"] = view_out["pooled"]
            out["view_structural"] = view_out["structural"]
            out["view_quantized"] = view_out["quantized"]
            out["view_stages"] = view_out["stages"]
            out["view_expected_depth"] = view_out["expected_depth"]
            out["view_gate_mean"] = view_out["struct_gate_mean"]

        return out

    def _base_losses(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = batch["x"]
        x_view = batch.get("x_view")
        transform = batch.get("transform")

        recon = mse(out["recon"], x)
        kl = kl_divergence(out["mu"], out["logvar"])

        equiv = self._zero(x)
        inv = self._zero(x)
        scale = self._zero(x)
        spectral = self._zero(x)

        if x_view is not None:
            if self.use_action and transform is not None:
                equiv = mse(out["action_latent"], out["view_latent"].detach())
                equiv = equiv + 0.2 * mse(
                    out["struct_gate_mean"],
                    out["view_gate_mean"].detach(),
                )
            else:
                equiv = mse(out["latent"], out["view_latent"].detach())

            inv = mse(out["pooled"], out["view_pooled"].detach())
            inv = inv + cosine_distance(
                out["quantized"], out["view_quantized"].detach()
            )

            spectral = spectral_distance(
                out["stages"][-1],
                out["view_stages"][-1].detach(),
            )
            spectral = spectral + spectral_distance(
                out["pooled"], out["view_pooled"].detach()
            )

            scale = scale + sum(
                mse(a, b.detach()) for a, b in zip(out["stages"], out["view_stages"])
            ) / max(1, len(out["stages"]))

        for a, b in zip(out["stages"][:-1], out["stages"][1:]):
            scale = scale + mse(a, b)

        geom = self._zero(x)
        align = self._zero(x)

        for i in range(self.num_levels):
            geom = geom + self._stack_stat(out, f"renorm_{i}_orthogonality", x)
            geom = geom + self._stack_stat(out, f"renorm_{i}_idempotence", x)
            align = align + self._stack_stat(out, f"renorm_{i}_alignment", x)

        proj = self._zero(x)
        for i in range(self.num_levels):
            proj = proj + self._stack_stat(out, f"renorm_{i}_projection_energy", x)
            proj = proj + 0.5 * self._stack_stat(
                out, f"renorm_{i}_projection_residual", x
            )

        node_flat = out["nodes"].reshape(x.shape[0], -1)
        node_entropy_floor = torch.log(
            self._num_nodes_tensor.to(device=x.device, dtype=x.dtype)
        )
        node = out["node_separation"]
        node = node + variance_penalty(node_flat)
        node = node + covariance_penalty(node_flat)
        node = node + torch.relu(node_entropy_floor - out["node_entropy"])
        node = node + out["edge_sparsity"]

        vq = out["vq_commit"] + out["vq_code"] + torch.relu(1.0 - out["vq_usage"])

        dispersion = out["structural"].std(dim=0, unbiased=False).mean()
        moment = variance_penalty(out["structural"])
        moment = moment + covariance_penalty(out["structural"])
        moment = moment + torch.relu(0.1 - dispersion)

        if self.use_action and transform is not None:
            identity = mse(out["identity_latent"], out["latent"])
        else:
            identity = self._zero(x)

        depth = (
            out["expected_depth"]
            - self._depth_target.to(device=x.device, dtype=x.dtype)
        ).abs()
        depth = depth + torch.relu(0.15 - out["stack_depth"])

        geom = geom + align

        if not self.use_equivariance_loss:
            equiv = self._zero(x)
        if not self.use_spectral_loss:
            spectral = self._zero(x)
        if not self.use_identity_loss:
            identity = self._zero(x)
        if not self.use_depth_loss:
            depth = self._zero(x)

        if not self.use_projective:
            proj = self._zero(x)
            geom = self._zero(x)
            spectral = self._zero(x)
            depth = self._zero(x)

        if not self.use_quantizer:
            vq = self._zero(x)

        return {
            "recon": recon,
            "equiv": equiv if self.use_action else self._zero(x),
            "inv": inv,
            "scale": scale,
            "proj": proj,
            "spectral": spectral if self.use_projective else self._zero(x),
            "geom": geom,
            "vq": vq,
            "node": node,
            "moment": moment,
            "identity": identity,
            "kl": kl,
            "depth": depth if self.use_projective else self._zero(x),
        }

    def losses(
        self,
        batch: dict[str, torch.Tensor],
        weights: dict[str, float],
        warmup: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = batch["x"]
        transform = batch.get("transform")
        x_view = batch.get("x_view")

        out = self.forward(x, x_view=x_view, transform=transform)
        base = self._base_losses(out, batch)

        effective_weights = dict(weights)
        if "rec" in effective_weights and "recon" not in effective_weights:
            effective_weights["recon"] = effective_weights.pop("rec")

        # Mechanism flags also disable their uncertainty offsets, even when a
        # custom config retains positive weights for an unavailable objective.
        active = {
            "proj": self.use_projective,
            "geom": self.use_projective,
            "spectral": self.use_projective and self.use_spectral_loss,
            "depth": self.use_projective and self.use_depth_loss,
            "vq": self.use_quantizer,
            "equiv": self.use_action
            and self.use_equivariance_loss
            and x_view is not None,
            "identity": self.use_action
            and self.use_identity_loss
            and transform is not None,
        }
        for key, enabled in active.items():
            if not enabled:
                effective_weights[key] = 0.0

        if warmup:
            for key in [
                "equiv",
                "inv",
                "scale",
                "proj",
                "spectral",
                "geom",
                "vq",
                "node",
                "moment",
                "identity",
                "depth",
            ]:
                base[key] = self._zero(x)
                effective_weights[key] = 0.0

        total, bal_terms = self.loss_balancer(base, effective_weights)

        losses = {**base, **bal_terms, "total": total}
        losses.update(
            {
                "perplexity_coarse": out["vq_coarse_perplexity"],
                "perplexity_fine": out["vq_fine_perplexity"],
                "usage": out["vq_usage"],
                "node_entropy": out["node_entropy"],
                "edge_entropy": out["edge_entropy"],
                "depth_mean": out["expected_depth"],
                "stack_geodesic": out["stack_geodesic"],
                "stack_alignment": out["stack_alignment"],
                "graph_energy": out["graph_energy"],
                "effective_rank": out["effective_rank"],
                "struct_gate_mean": out["struct_gate_mean"],
                "view_gate_mean": out.get("view_gate_mean", out["struct_gate_mean"]),
            }
        )
        return losses
