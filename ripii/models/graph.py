from __future__ import annotations

import torch
from torch import nn


class GraphMessageBlock(nn.Module):
    def __init__(self, node_dim: int, edge_types: int = 3) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_types = int(edge_types)
        pair_dim = node_dim * 4
        self.edge_logits = nn.Sequential(
            nn.Linear(pair_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
        )
        self.edge_type = nn.Sequential(
            nn.Linear(pair_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, edge_types),
        )
        self.message = nn.Sequential(
            nn.Linear(pair_dim, node_dim * edge_types),
            nn.GELU(),
            nn.Linear(node_dim * edge_types, node_dim * edge_types),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, nodes: torch.Tensor, topk: int = 2
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, n, d = nodes.shape
        left = nodes.unsqueeze(2).expand(b, n, n, d)
        right = nodes.unsqueeze(1).expand(b, n, n, d)
        pair = torch.cat([left, right, left - right, left * right], dim=-1)
        logits = self.edge_logits(pair).squeeze(-1)
        if n > 1:
            diagonal = torch.eye(n, device=nodes.device, dtype=torch.bool)
            logits = logits.masked_fill(diagonal.unsqueeze(0), float("-inf"))
        dense_adj = torch.softmax(logits, dim=-1)
        active_neighbors = n if n == 1 else n - 1
        if topk and topk > 0 and topk < active_neighbors:
            active_neighbors = topk
            _, topi = torch.topk(logits, k=active_neighbors, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, topi, True)
            logits = logits.masked_fill(~mask, float("-inf"))
            sparse_adj = torch.softmax(logits, dim=-1)
            # Sparse values are used in the forward pass while dense attention
            # supplies a straight-through gradient for the discrete top-k choice.
            adj = sparse_adj + dense_adj - dense_adj.detach()
        else:
            adj = dense_adj
        types = torch.softmax(self.edge_type(pair), dim=-1)
        msgs = self.message(pair).view(b, n, n, self.edge_types, d)
        typed = torch.sum(types.unsqueeze(-1) * msgs, dim=-2)
        agg = torch.sum(adj.unsqueeze(-1) * typed, dim=2)
        upd = self.update(torch.cat([nodes, agg], dim=-1))
        gate = self.node_gate(nodes)
        out = nodes + self.scale * gate * self.norm(upd)
        sims = torch.nn.functional.cosine_similarity(
            nodes.unsqueeze(2), nodes.unsqueeze(1), dim=-1
        )
        tri = torch.triu(
            torch.ones(n, n, device=nodes.device, dtype=torch.bool), diagonal=1
        )
        sep = (
            sims.masked_select(tri).abs().mean()
            if tri.any()
            else torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        )
        ent = -(adj * adj.clamp_min(1e-9).log()).sum(dim=-1).mean()
        support = adj > 0
        sparse = support.float().mean()
        degree = support.float().sum(dim=-1).mean()
        if active_neighbors > 1:
            normalizer = torch.log(
                torch.tensor(
                    float(active_neighbors), device=nodes.device, dtype=nodes.dtype
                )
            )
            concentration_loss = ent / normalizer
        else:
            concentration_loss = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        self_edge_mass = torch.diagonal(adj, dim1=-2, dim2=-1).mean()
        stats = {
            "node_separation": sep,
            "edge_entropy": ent,
            "edge_sparsity": 1.0 - sparse,
            "edge_concentration_loss": concentration_loss,
            "avg_degree": degree,
            "self_edge_mass": self_edge_mass,
            "graph_energy": out.pow(2).mean(),
        }
        return out, stats


class LatentGraphModule(nn.Module):
    def __init__(self, node_dim: int, steps: int, topk: int = 2) -> None:
        super().__init__()
        self.steps = max(0, int(steps))
        self.topk = int(topk)
        self.blocks = nn.ModuleList(
            [GraphMessageBlock(node_dim) for _ in range(self.steps)]
        )
        self.readout = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.GELU(), nn.Linear(node_dim, node_dim)
        )
        self.attn = nn.Linear(node_dim, 1)
        self.post = nn.LayerNorm(node_dim)

    def forward(
        self, nodes: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.steps == 0:
            pooled = nodes.mean(dim=1)
            entropy = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
            empty_stats = {
                "node_separation": torch.zeros(
                    (), device=nodes.device, dtype=nodes.dtype
                ),
                "edge_entropy": torch.zeros((), device=nodes.device, dtype=nodes.dtype),
                "edge_sparsity": torch.zeros(
                    (), device=nodes.device, dtype=nodes.dtype
                ),
                "edge_concentration_loss": torch.zeros(
                    (), device=nodes.device, dtype=nodes.dtype
                ),
                "avg_degree": torch.zeros((), device=nodes.device, dtype=nodes.dtype),
                "self_edge_mass": torch.zeros(
                    (), device=nodes.device, dtype=nodes.dtype
                ),
                "node_entropy": entropy,
                "graph_energy": pooled.pow(2).mean(),
            }
            return pooled, empty_stats
        stats: dict[str, torch.Tensor] = {}
        sep_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        ent_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        sparse_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        concentration_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        deg_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        self_edge_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        for idx, block in enumerate(self.blocks):
            nodes, block_stats = block(nodes, self.topk)
            for key, value in block_stats.items():
                stats[f"graph_{idx}_{key}"] = value
            sep_total = sep_total + block_stats["node_separation"]
            ent_total = ent_total + block_stats["edge_entropy"]
            sparse_total = sparse_total + block_stats["edge_sparsity"]
            concentration_total = (
                concentration_total + block_stats["edge_concentration_loss"]
            )
            deg_total = deg_total + block_stats["avg_degree"]
            self_edge_total = self_edge_total + block_stats["self_edge_mass"]
        read = self.post(self.readout(nodes))
        logits = self.attn(read).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * read, dim=1)
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
        stats.update(
            {
                "node_separation": sep_total / len(self.blocks),
                "edge_entropy": ent_total / len(self.blocks),
                "edge_sparsity": sparse_total / len(self.blocks),
                "edge_concentration_loss": concentration_total / len(self.blocks),
                "avg_degree": deg_total / len(self.blocks),
                "self_edge_mass": self_edge_total / len(self.blocks),
                "node_entropy": entropy,
                "graph_energy": read.pow(2).mean(),
            }
        )
        return pooled, stats
