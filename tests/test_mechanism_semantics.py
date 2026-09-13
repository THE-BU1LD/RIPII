from __future__ import annotations

import torch

from ripii.models.graph import GraphMessageBlock
from ripii.models.projective import ProjectiveRenormStack, SubspaceProjector
from ripii.models.quantizer import HierarchicalVectorQuantizer
from ripii.models.ripii import RIPIIModel
from ripii.utils.ablation import apply_mode
from ripii.utils.config import Config


def test_num_projectors_controls_capacity_and_selector_shape():
    one = ProjectiveRenormStack(dim=12, num_levels=2, num_projectors=1)
    three = ProjectiveRenormStack(dim=12, num_levels=2, num_projectors=3)
    assert len(one.projectors) == 1
    assert len(three.projectors) == 3
    assert one.selectors[0].out_features == 1
    assert three.selectors[0].out_features == 3
    assert sum(p.numel() for p in three.parameters()) > sum(
        p.numel() for p in one.parameters()
    )
    disabled = ProjectiveRenormStack(dim=12, num_levels=0, num_projectors=9)
    assert not list(disabled.projectors)


def test_raw_basis_orthogonality_is_nontrivial_and_differentiable():
    projector = SubspaceProjector(dim=8, rank=3)
    with torch.no_grad():
        projector.raw_basis[:, 1].copy_(projector.raw_basis[:, 0])
    _, stats, _ = projector(torch.randn(5, 8))
    assert stats["orthogonality"] > 0.1
    gradient = torch.autograd.grad(stats["orthogonality"], projector.raw_basis)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_projection_objective_minimizes_residual_not_captured_energy():
    model = RIPIIModel(24, 12, 24, 4, 3, 1, 1, 4, 8, 1)
    batch = {
        "x": torch.randn(6, 24),
        "x_view": torch.randn(6, 24),
        "transform": torch.randn(6, 4),
    }
    out = model.forward(
        batch["x"], x_view=batch["x_view"], transform=batch["transform"]
    )
    losses = model._base_losses(out, batch)
    assert torch.allclose(losses["proj"], out["renorm_0_projection_residual"])


def test_graph_excludes_self_edges_and_uses_differentiable_concentration():
    block = GraphMessageBlock(node_dim=6)
    _, stats = block(torch.randn(2, 5, 6), topk=2)
    assert torch.allclose(
        stats["self_edge_mass"], torch.zeros_like(stats["self_edge_mass"])
    )
    assert stats["avg_degree"] == 2
    assert stats["edge_sparsity"] == 0.6
    assert stats["edge_concentration_loss"].requires_grad
    edge_parameters = list(block.edge_logits.parameters())
    gradients = torch.autograd.grad(
        stats["edge_concentration_loss"], edge_parameters, allow_unused=True
    )
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)


def test_top_one_graph_selection_has_straight_through_edge_gradients():
    block = GraphMessageBlock(node_dim=6)
    output, _ = block(torch.randn(2, 5, 6), topk=1)
    gradients = torch.autograd.grad(
        output.square().mean(), block.edge_logits.parameters()
    )
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_vq_balance_objective_reaches_inputs_and_codebooks():
    quantizer = HierarchicalVectorQuantizer(8, 8, 6)
    inputs = torch.randn(16, 6, requires_grad=True)
    _, stats = quantizer(inputs)
    assert not stats["vq_usage"].requires_grad
    assert stats["vq_balance"].requires_grad
    input_grad, coarse_grad, fine_grad = torch.autograd.grad(
        stats["vq_balance"], (inputs, quantizer.coarse, quantizer.fine)
    )
    assert input_grad.abs().sum() > 0
    assert coarse_grad.abs().sum() > 0
    assert fine_grad.abs().sum() > 0


def test_dead_code_revival_is_explicit_deterministic_and_finite():
    torch.manual_seed(71)
    values = torch.randn(32, 4)
    first = HierarchicalVectorQuantizer(8, 8, 4)
    second = HierarchicalVectorQuantizer(8, 8, 4)
    with torch.no_grad():
        first.coarse.zero_()
        first.fine.zero_()
        second.load_state_dict(first.state_dict())
    result_a = first.revive_dead_codes(values)
    result_b = second.revive_dead_codes(values)
    assert result_a == result_b
    assert result_a["coarse_codes_reset"] >= 7
    assert result_a["fine_codes_reset"] >= 7
    assert torch.equal(first.coarse, second.coarse)
    assert torch.equal(first.fine, second.fine)
    assert torch.isfinite(first.coarse).all() and torch.isfinite(first.fine).all()


def test_direct_mechanism_sweep_modes_change_the_intended_axis():
    cfg = Config()
    assert apply_mode(cfg, "projectors_1").num_projectors == 1
    assert apply_mode(cfg, "levels_0").num_levels == 0
    assert apply_mode(cfg, "graph_topk_full").graph_topk == cfg.num_nodes - 1
    assert apply_mode(cfg, "no_vq_balance").use_vq_balance is False
