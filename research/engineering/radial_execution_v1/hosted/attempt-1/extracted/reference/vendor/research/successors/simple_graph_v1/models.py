"""Isolated continuous successors; historical RIPII source remains unchanged."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ripii.world.models import Interaction, WorldModel, mlp

ROOT = Path(__file__).resolve().parent
MODELS = ("simple_pool", "graph", "full_graph", "transformer", "multiscale", "nri")


class HistoryEncoder(nn.Module):
    """All graph comparators receive the same observed history, never targets."""

    def __init__(self, hidden, context):
        super().__init__()
        self.history = mlp(context * 4, hidden, hidden)
        self.current = mlp(8 + hidden, hidden, hidden)
        self.embedding = None

    def observe(self, history):
        # [batch, observed time, object, position/velocity]
        self.embedding = self.history(history.transpose(1, 2).flatten(2))

    def forward(self, inputs):
        if self.embedding is None:
            raise RuntimeError("Observed history must be supplied before prediction")
        return self.current(torch.cat((inputs, self.embedding), dim=-1))


class FullInteraction(Interaction):
    def forward(self, h, position, velocity, mask, local=True):
        return super().forward(h, position, velocity, mask, local=False)


class ContinuousGraph(nn.Module):
    """Winning global-pool backbone with continuous state and prediction loss only."""

    def __init__(self, name, hidden=64, context=8):
        super().__init__()
        variant = {"simple_pool": "global_pool", "full_graph": "graph"}.get(name, name)
        self.backbone = WorldModel(
            variant, hidden, max_objects=5, bottleneck="continuous"
        )
        self.backbone.encoder = HistoryEncoder(hidden, context)
        if name == "full_graph":
            self.backbone.local = FullInteraction(hidden)
            self.backbone.refine = FullInteraction(hidden)

    def forward(self, history, horizon):
        self.backbone.encoder.observe(history)
        current = history[:, -1]
        properties = torch.ones_like(current[..., :2])
        properties[..., 0] = 0.04
        state = torch.cat((current, properties), dim=-1)
        mask = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        action = torch.zeros_like(state[..., :2])
        predictions = []
        for _ in range(horizon):
            state = self.backbone(state, action, mask)
            predictions.append(state[..., :4])
        return torch.stack(predictions, dim=1), state.new_zeros(())


def upstream_modules():
    # Preserve upstream bytes. Its absolute `from utils` is scoped to import only.
    old = sys.modules.pop("utils", None)
    try:
        for name in ("utils", "modules"):
            spec = importlib.util.spec_from_file_location(
                "utils" if name == "utils" else "simple_graph_nri_modules",
                ROOT / "upstream" / f"{name}.py",
            )
            module = importlib.util.module_from_spec(spec)
            if name == "utils":
                sys.modules["utils"] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("utils", None)
        if old is not None:
            sys.modules["utils"] = old


class OfficialNRI(nn.Module):
    """Pinned official encoder/decoder, adapted to common context/forecast contract.

    This is an adapted official-code baseline, not reproduction of paper scores.
    No ground-truth edges, future frames, or test labels enter the encoder.
    """

    def __init__(self, hidden=64, context=8):
        super().__init__()
        upstream = upstream_modules()
        self.encoder = upstream.MLPEncoder(context * 4, hidden, 2, do_prob=0.0)
        self.decoder = upstream.MLPDecoder(4, 2, hidden, hidden, hidden, do_prob=0.0)
        receiver, sender = (~torch.eye(5, dtype=torch.bool)).nonzero(as_tuple=True)
        self.register_buffer("receivers", F.one_hot(receiver, 5).float())
        self.register_buffer("senders", F.one_hot(sender, 5).float())

    def forward(self, history, horizon):
        logits = self.encoder(
            history.transpose(1, 2).contiguous(), self.receivers, self.senders
        )
        probabilities = logits.softmax(-1)
        if self.training:
            edges = F.gumbel_softmax(logits, tau=0.5, hard=True, dim=-1)
        else:
            edges = F.one_hot(logits.argmax(-1), 2).to(logits.dtype)
        current = history[:, -1:].contiguous()
        predictions = []
        for _ in range(horizon):
            current = self.decoder.single_step_forward(
                current, self.receivers, self.senders, edges[:, None]
            )
            predictions.append(current[:, 0])
        # Sum over directed edges/classes, mean over scenes, divide by node count.
        kl = (
            probabilities
            * (probabilities.clamp_min(1e-12).log() + torch.log(logits.new_tensor(2.0)))
        ).sum((1, 2)).mean() / 5
        return torch.stack(predictions, 1), kl


def make_model(name, hidden=64, context=8):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name}")
    return (
        OfficialNRI(hidden, context)
        if name == "nri"
        else ContinuousGraph(name, hidden, context)
    )
