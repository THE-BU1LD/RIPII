from __future__ import annotations

import copy

import pytest
import torch

from ripii.models.quantizer import HierarchicalVectorQuantizer
from ripii.models.ripii import RIPIIModel
from ripii.models.stable_quantizer import StableHierarchicalVectorQuantizer


def test_cancellation_witness_selects_the_actual_nearest_code():
    old = HierarchicalVectorQuantizer(2, 1, 1)
    new = StableHierarchicalVectorQuantizer(old)
    x = torch.tensor([[10001.0]])
    codebook = torch.tensor([[10000.0], [10001.0]])
    oracle = (x.double()[:, None] - codebook.double()[None]).square().sum(-1)
    assert old._pairwise_distance(x, codebook).argmin(-1).item() == 0
    distance = new._pairwise_distance(x, codebook)
    torch.testing.assert_close(distance.double(), oracle, rtol=0, atol=0)
    assert new._quantize(x, codebook)[1].item() == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("n,k,d", [(1, 1, 1), (7, 5, 3), (31, 29, 6)])
def test_direct_float64_oracle_and_assignment(dtype, n, k, d):
    gen = torch.Generator().manual_seed(33)
    x = (torch.randn(n, d, generator=gen) * 2).to(dtype)
    code = (torch.randn(k, d, generator=gen) * 2).to(dtype)
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(k, 2, d))
    oracle = (x.double()[:, None] - code.double()[None]).square().sum(-1)
    actual = wrapper._pairwise_distance(x, code)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(actual.double(), oracle, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.argmin(-1), oracle.argmin(-1))


def test_distance_gradients_match_direct_subtraction_and_gradcheck():
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(4, 2, 3))
    gen = torch.Generator().manual_seed(47)
    x = torch.randn(5, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    code = torch.randn(4, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    actual = wrapper._pairwise_distance(x, code)
    oracle = (x[:, None] - code[None]).square().sum(-1)
    ag = torch.autograd.grad((actual * weight).sum(), (x, code), retain_graph=True)
    rg = torch.autograd.grad((oracle * weight).sum(), (x, code), retain_graph=True)
    for a, r in zip(ag, rg, strict=True):
        torch.testing.assert_close(a, r, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(wrapper._pairwise_distance, (x, code))


def test_exact_ties_retain_first_index_and_zero_distance_gradient():
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(3, 1, 1))
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    code = torch.tensor([[-1.0], [1.0], [1.0]], dtype=torch.float64, requires_grad=True)
    distance = wrapper._pairwise_distance(x, code)
    assert distance.argmin(-1).tolist() == [0, 1]
    gx, gc = torch.autograd.grad(distance[1, 1], (x, code))
    assert torch.equal(gx, torch.zeros_like(gx))
    assert torch.equal(gc, torch.zeros_like(gc))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cpu_autocast_does_not_downcast_distance_work(dtype):
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(2, 1, 1))
    x = torch.tensor([[10001.0]])
    code = torch.tensor([[10000.0], [10001.0]])
    with torch.autocast("cpu", dtype=dtype):
        distance = wrapper._pairwise_distance(x, code)
    assert distance.dtype == torch.float32
    assert distance.tolist() == [[1.0, 0.0]]


def test_wrapper_preserves_parameters_rng_mode_and_checkpoint():
    old = HierarchicalVectorQuantizer(4, 3, 2).double().eval()
    optimizer = torch.optim.SGD(old.parameters(), lr=0.1)
    rng = torch.random.get_rng_state().clone()
    wrapper = StableHierarchicalVectorQuantizer(old)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert wrapper.coarse is old.coarse and wrapper.fine is old.fine
    assert optimizer.param_groups[0]["params"][0] is wrapper.coarse
    assert not wrapper.training
    assert list(wrapper.state_dict()) == list(old.state_dict())
    wrapper.load_state_dict(old.state_dict(), strict=True)
    old.load_state_dict(wrapper.state_dict(), strict=True)


def test_full_quantizer_outputs_losses_and_gradients_match_direct_reference():
    class DirectReference(HierarchicalVectorQuantizer):
        def _pairwise_distance(self, x, codebook):
            return (x[:, None] - codebook[None]).square().sum(-1)

    torch.manual_seed(32)
    reference = DirectReference(5, 4, 3).double()
    wrapper = StableHierarchicalVectorQuantizer(copy.deepcopy(reference))
    a = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    actual, stats = wrapper(a)
    expected, expected_stats = reference(b)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert stats.keys() == expected_stats.keys()
    for name in stats:
        torch.testing.assert_close(stats[name], expected_stats[name], rtol=1e-10, atol=1e-12)
    al = actual.square().mean() + stats["vq_commit"] + stats["vq_code"] + stats["vq_balance"]
    rl = expected.square().mean() + expected_stats["vq_commit"] + expected_stats["vq_code"] + expected_stats["vq_balance"]
    ag = torch.autograd.grad(al, (a, wrapper.coarse, wrapper.fine))
    rg = torch.autograd.grad(rl, (b, reference.coarse, reference.fine))
    for x, y in zip(ag, rg, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-12)


def test_real_model_forward_backward_and_optimizer_step():
    torch.manual_seed(72)
    model = RIPIIModel(24, 12, 24, 4, 3, 1, 1, 4, 8, 1)
    model.quantizer = StableHierarchicalVectorQuantizer(model.quantizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.quantizer.coarse.detach().clone()
    x = torch.randn(6, 24)
    batch = {"x": x, "x_view": torch.randn_like(x), "transform": torch.randn(6, 4)}
    losses = model.losses(batch, {"recon": 1.0, "vq": 1.0}, warmup=False)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.quantizer.coarse.grad is not None
    assert torch.isfinite(model.quantizer.coarse.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.quantizer.coarse)


@pytest.mark.parametrize("shape", [(), (0, 2), (2, 0), (3,), (2, 3)])
def test_invalid_forward_shape_rejected(shape):
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(3, 2, 2))
    with pytest.raises(ValueError, match="dimension"):
        wrapper(torch.empty(shape))


@pytest.mark.parametrize("value", [torch.tensor([1, 2]), torch.tensor([1j, 2j]), [1.0, 2.0]])
def test_nonfloating_forward_rejected(value):
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(3, 2, 2))
    with pytest.raises(TypeError, match="floating"):
        wrapper(value)


@pytest.mark.parametrize("which", ["input", "codebook"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_distance_operand_rejected(which, value):
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(3, 2, 2))
    x, code = torch.zeros(1, 2), torch.zeros(3, 2)
    (x if which == "input" else code)[0, 0] = value
    with pytest.raises(ValueError, match=which):
        wrapper._pairwise_distance(x, code)


def test_overflow_is_rejected_instead_of_an_all_infinite_tie():
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(2, 1, 1))
    x = torch.tensor([[1e30]])
    code = torch.tensor([[-1e30], [0.0]])
    before = wrapper.coarse.detach().clone()
    with pytest.raises(FloatingPointError, match="distances"):
        wrapper._pairwise_distance(x, code)
    assert torch.equal(before, wrapper.coarse)


@pytest.mark.parametrize("dtype,value", [(torch.float32, 1e-30), (torch.float64, 1e-200)])
def test_underflow_cannot_create_a_false_exact_match(dtype, value):
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(2, 1, 1))
    x = torch.tensor([[value]], dtype=dtype)
    code = torch.tensor([[0.0], [value]], dtype=dtype)
    assert x[0, 0] != code[0, 0]
    with pytest.raises(FloatingPointError, match="underflowed"):
        wrapper._quantize(x, code)


def test_noncontiguous_inputs_and_scalar_vector():
    wrapper = StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(3, 2, 2)).double()
    x = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2).transpose(0, 1)
    assert not x.is_contiguous()
    out, _ = wrapper(x)
    assert out.shape == x.shape
    assert wrapper(torch.tensor([0.1, 0.2], dtype=torch.float64))[0].shape == (2,)


def test_zero_code_dimension_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer(2, 1, 0))
