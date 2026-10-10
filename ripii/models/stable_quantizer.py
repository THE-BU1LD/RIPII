"""Opt-in numerical successor; retained quantizer source and callers stay fixed."""

from __future__ import annotations

import torch
from torch import nn

from .quantizer import HierarchicalVectorQuantizer


class StableHierarchicalVectorQuantizer(HierarchicalVectorQuantizer):
    """Reuse a quantizer's parameters with direct Euclidean distance evaluation.

    The retained norm/matmul identity loses low-order bits for nearby vectors
    with large common offsets. This explicit wrapper changes only the distance
    kernel. All inherited VQ, straight-through and loss equations remain in
    force. It shares the original Parameters, so existing optimizers and strict
    state dictionaries remain compatible and construction uses no RNG draws.

    Distance work is float64 when either operand is float64, otherwise float32;
    half/bfloat16 inputs are promoted. Autocast cannot lower that work precision.
    Non-finite inputs or distances raise instead of selecting a spurious first
    code from an all-infinite tie. Finite precision can still create true ties;
    the inherited argmin retains its first-code rule. This is not a promise of
    bitwise equivalence, speed improvement, or efficacy on historical studies.
    """

    def __init__(self, quantizer: HierarchicalVectorQuantizer) -> None:
        if not isinstance(quantizer, HierarchicalVectorQuantizer):
            raise TypeError("expected a HierarchicalVectorQuantizer")
        if quantizer.code_dim < 1:
            raise ValueError("code dimension must be positive")
        nn.Module.__init__(self)
        self.num_coarse = quantizer.num_coarse
        self.num_fine = quantizer.num_fine
        self.code_dim = quantizer.code_dim
        self.beta = quantizer.beta
        self.coarse = quantizer.coarse
        self.fine = quantizer.fine
        self.train(quantizer.training)

    def _pairwise_distance(
        self, x: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        for name, value in (("input", x), ("codebook", codebook)):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a real floating tensor")
            if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != self.code_dim:
                raise ValueError(f"{name} must have nonempty shape (N, {self.code_dim})")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if x.device != codebook.device:
            raise ValueError("input and codebook must be on the same device")
        dtype = (
            torch.float64
            if torch.float64 in (x.dtype, codebook.dtype)
            else torch.float32
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_work, code_work = x.to(dtype=dtype), codebook.to(dtype=dtype)
            distances = torch.cdist(
                x_work,
                code_work,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).square()
        if not torch.isfinite(distances).all():
            raise FloatingPointError("squared code distances are not finite at the work precision")
        # A represented nonzero separation can also underflow to zero. Never
        # allow that to become a false exact-match/first-code assignment. Check
        # all zero pairs in bounded feature chunks without an N-by-K-by-D tensor.
        rows, columns = (distances == 0).nonzero(as_tuple=True)
        chunk = max(1, 65536 // self.code_dim)
        for start in range(0, rows.numel(), chunk):
            stop = start + chunk
            if (x_work[rows[start:stop]] != code_work[columns[start:stop]]).any():
                raise FloatingPointError("nonzero code distance underflowed to zero")
        return distances

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("input must be a real floating tensor")
        if x.ndim < 1 or x.shape[-1] != self.code_dim or x.numel() == 0:
            raise ValueError(f"input must have nonempty trailing dimension {self.code_dim}")
        return super().forward(x)
