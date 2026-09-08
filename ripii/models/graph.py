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
        if topk and topk > 0 and topk < n:
            _, topi = torch.topk(logits, k=topk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, topi, True)
            logits = logits.masked_fill(~mask, float("-inf"))
        adj = torch.softmax(logits, dim=-1)
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
        sparse = (adj > 0).float().mean()
        degree = adj.sum(dim=-1).mean()
        stats = {
            "node_separation": sep,
            "edge_entropy": ent,
            "edge_sparsity": 1.0 - sparse,
            "avg_degree": degree,
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
            stats = {
                "node_separation": torch.zeros(
                    (), device=nodes.device, dtype=nodes.dtype
                ),
                "edge_entropy": torch.zeros((), device=nodes.device, dtype=nodes.dtype),
                "edge_sparsity": torch.zeros(
                    (), device=nodes.device, dtype=nodes.dtype
                ),
                "avg_degree": torch.zeros((), device=nodes.device, dtype=nodes.dtype),
                "node_entropy": entropy,
                "graph_energy": pooled.pow(2).mean(),
            }
            return pooled, stats
        stats: dict[str, torch.Tensor] = {}
        sep_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        ent_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        sparse_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        deg_total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        for idx, block in enumerate(self.blocks):
            nodes, block_stats = block(nodes, self.topk)
            for key, value in block_stats.items():
                stats[f"graph_{idx}_{key}"] = value
            sep_total = sep_total + block_stats["node_separation"]
            ent_total = ent_total + block_stats["edge_entropy"]
            sparse_total = sparse_total + block_stats["edge_sparsity"]
            deg_total = deg_total + block_stats["avg_degree"]
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
                "avg_degree": deg_total / len(self.blocks),
                "node_entropy": entropy,
                "graph_energy": read.pow(2).mean(),
            }
        )
        return pooled, stats
