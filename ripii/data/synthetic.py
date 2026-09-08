from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class StructuralSample:
    x: torch.Tensor
    x_view: torch.Tensor
    transform: torch.Tensor
    label: torch.Tensor
    orbit: torch.Tensor
    motif: torch.Tensor
    complexity: torch.Tensor


def _orthonormal_basis(
    generator: torch.Generator, components: int, dim: int
) -> torch.Tensor:
    components = max(1, min(int(components), int(dim)))
    basis, _ = torch.linalg.qr(
        torch.randn(dim, dim, generator=generator), mode="reduced"
    )
    return basis[:, :components].T.contiguous()


class SyntheticStructuralDataset(Dataset[StructuralSample]):
    def __init__(
        self,
        size: int,
        input_dim: int,
        noise_std: float,
        transform_scale: float,
        transform_shift: int,
        transform_dim: int = 4,
        num_classes: int = 8,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.input_dim = int(input_dim)
        self.noise_std = float(noise_std)
        self.transform_scale = float(transform_scale)
        self.transform_shift = int(transform_shift)
        self.transform_dim = int(transform_dim)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        gen = torch.Generator().manual_seed(self.seed)
        semantic_dim = max(8, self.num_classes + 4)
        style_dim = max(6, self.input_dim // 8)
        nuisance_dim = max(6, self.input_dim // 10)
        orbit_count = max(2, min(6, max(1, self.size // max(1, self.num_classes))))
        motif_count = max(4, min(8, self.num_classes + 2))
        self.semantic_basis = _orthonormal_basis(gen, semantic_dim, self.input_dim)
        self.style_basis = _orthonormal_basis(gen, style_dim, self.input_dim)
        self.nuisance_basis = _orthonormal_basis(gen, nuisance_dim, self.input_dim)
        self.class_embeddings = (
            torch.randn(self.num_classes, semantic_dim, generator=gen) * 0.7
        )
        self.orbit_embeddings = (
            torch.randn(orbit_count, semantic_dim, generator=gen) * 0.4
        )
        self.motif_basis = torch.randn(motif_count, semantic_dim, generator=gen) * 0.45
        self.scale_templates = torch.randn(4, self.input_dim, generator=gen) * 0.08
        self.samples = [self._make_sample(i, gen) for i in range(self.size)]

    def _semantic_state(
        self, idx: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        orbit = torch.tensor(
            (idx // max(1, self.num_classes)) % self.orbit_embeddings.shape[0],
            dtype=torch.long,
        )
        motif = torch.tensor(idx % self.motif_basis.shape[0], dtype=torch.long)
        complexity = torch.tensor((idx % 5) + 1, dtype=torch.long)
        semantic = self.class_embeddings[label].clone()
        semantic = semantic + self.orbit_embeddings[orbit]
        semantic = semantic + self.motif_basis[motif]
        semantic = semantic + 0.05 * torch.randn(semantic.shape, generator=generator)
        return semantic, label, orbit, motif, complexity

    def _style_state(self, generator: torch.Generator) -> torch.Tensor:
        coeffs = torch.randn(self.style_basis.shape[0], generator=generator)
        return coeffs / coeffs.norm().clamp_min(1e-6)

    def _nuisance_state(self, generator: torch.Generator) -> torch.Tensor:
        coeffs = torch.randn(self.nuisance_basis.shape[0], generator=generator)
        return coeffs / coeffs.norm().clamp_min(1e-6)

    def _render(
        self,
        semantic: torch.Tensor,
        style: torch.Tensor,
        nuisance: torch.Tensor,
        scale_id: int,
    ) -> torch.Tensor:
        x = semantic @ self.semantic_basis
        x = x + style @ self.style_basis
        x = x + nuisance @ self.nuisance_basis
        x = x + self.scale_templates[scale_id % self.scale_templates.shape[0]]
        return torch.tanh(x)

    def _transform(self, x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.transform_scale * transform[0]
        shift = int(abs(float(transform[1])) * self.transform_shift) % max(
            1, self.input_dim
        )
        flip = bool(transform[2] > 0)
        warp = transform[3]
        y = torch.roll(x, shifts=shift, dims=-1)
        if flip:
            y = torch.flip(y, dims=[-1])
        y = y * scale
        freqs = torch.linspace(
            1.0, 3.0, steps=self.input_dim, dtype=y.dtype, device=y.device
        )
        # Zero is the identity transform, including its warp component.
        y = y + 0.05 * (torch.sin(freqs * (warp + 1.0)) - torch.sin(freqs))
        return y

    def _make_sample(self, idx: int, generator: torch.Generator) -> StructuralSample:
        semantic, label, orbit, motif, complexity = self._semantic_state(idx, generator)
        style = self._style_state(generator)
        nuisance = self._nuisance_state(generator)
        scale_id = int(complexity.item() - 1)
        x = self._render(semantic, style, nuisance, scale_id)
        transform = torch.randn(self.transform_dim, generator=generator)
        x_view = self._transform(x, transform)
        x = x + self.noise_std * torch.randn(x.shape, generator=generator)
        x_view = x_view + self.noise_std * torch.randn(
            x_view.shape, generator=generator
        )
        return StructuralSample(
            x=x.float(),
            x_view=x_view.float(),
            transform=transform.float(),
            label=label,
            orbit=orbit,
            motif=motif,
            complexity=complexity,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> StructuralSample:
        return self.samples[idx]
