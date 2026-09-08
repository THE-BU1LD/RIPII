from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from ripii.models.quantizer import HierarchicalVectorQuantizer
from ripii.utils.metrics import sanitize
from ripii.utils.seed import seed_everything


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(
    *,
    samples: int,
    code_dim: int,
    coarse_codes: int,
    fine_codes: int,
    seed: int,
    centers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    if centers is None:
        basis, _ = torch.linalg.qr(torch.randn(code_dim, code_dim, generator=generator))
        coarse_centers = 3.0 * basis[:coarse_codes]
        fine_centers = 0.7 * torch.roll(basis, shifts=coarse_codes, dims=0)[:fine_codes]
    else:
        coarse_centers, fine_centers = centers
    coarse_labels = torch.arange(samples) % coarse_codes
    fine_labels = (torch.arange(samples) // coarse_codes) % fine_codes
    permutation = torch.randperm(samples, generator=generator)
    coarse_labels = coarse_labels[permutation]
    fine_labels = fine_labels[permutation]
    values = coarse_centers[coarse_labels] + fine_centers[fine_labels]
    values = values + 0.03 * torch.randn(values.shape, generator=generator)
    return values, coarse_labels, fine_labels, (coarse_centers, fine_centers)


def _purity(assignments: torch.Tensor, labels: torch.Tensor) -> float:
    correct = 0
    for assignment in assignments.unique():
        member_labels = labels[assignments == assignment]
        correct += int(torch.bincount(member_labels).max().item())
    return correct / max(1, labels.numel())


def run_qualification(
    *,
    seed: int,
    steps: int,
    samples: int,
    code_dim: int,
    coarse_codes: int,
    fine_codes: int,
    learning_rate: float,
) -> dict[str, Any]:
    if min(steps, samples, code_dim, coarse_codes, fine_codes) <= 0:
        raise ValueError("qualification dimensions and steps must be positive")
    if code_dim < coarse_codes + fine_codes:
        raise ValueError("code_dim must be at least coarse_codes + fine_codes")
    seed_everything(seed)
    train_x, _, _, centers = _dataset(
        samples=samples,
        code_dim=code_dim,
        coarse_codes=coarse_codes,
        fine_codes=fine_codes,
        seed=seed,
    )
    test_x, coarse_y, fine_y, _ = _dataset(
        samples=max(64, samples // 2),
        code_dim=code_dim,
        coarse_codes=coarse_codes,
        fine_codes=fine_codes,
        seed=seed + 1_000_003,
        centers=centers,
    )
    quantizer = HierarchicalVectorQuantizer(coarse_codes, fine_codes, code_dim)
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=learning_rate)
    for _ in range(steps):
        _, stats = quantizer(train_x)
        loss = stats["vq_commit"] + stats["vq_code"]
        sanitize(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in quantizer.parameters():
            if parameter.grad is not None:
                sanitize(parameter.grad)
        optimizer.step()

    with torch.no_grad():
        coarse_quant, coarse_idx, coarse_probs = quantizer._quantize(
            test_x, quantizer.coarse
        )
        residual = test_x - coarse_quant
        fine_quant, fine_idx, fine_probs = quantizer._quantize(residual, quantizer.fine)
        reconstruction_mse = float(F.mse_loss(coarse_quant + fine_quant, test_x).item())
        coarse_fraction = float((coarse_probs > 0).float().mean().item())
        fine_fraction = float((fine_probs > 0).float().mean().item())
        coarse_purity = _purity(coarse_idx, coarse_y)
        fine_purity = _purity(fine_idx, fine_y)

    gates = {
        "coarse_effective_fraction_at_least_0_5": coarse_fraction >= 0.5,
        "fine_effective_fraction_at_least_0_5": fine_fraction >= 0.5,
        "coarse_purity_at_least_0_8": coarse_purity >= 0.8,
        "fine_purity_at_least_0_8": fine_purity >= 0.8,
        "reconstruction_mse_at_most_0_1": reconstruction_mse <= 0.1,
    }
    return {
        "study": "isolated_hierarchical_quantizer_qualification",
        "evidence_status": "development_only",
        "seed": seed,
        "steps": steps,
        "train_samples": samples,
        "test_samples": int(test_x.shape[0]),
        "code_dim": code_dim,
        "coarse_codes": coarse_codes,
        "fine_codes": fine_codes,
        "metrics": {
            "coarse_effective_fraction": coarse_fraction,
            "fine_effective_fraction": fine_fraction,
            "coarse_assignment_purity": coarse_purity,
            "fine_assignment_purity": fine_purity,
            "reconstruction_mse": reconstruction_mse,
        },
        "gates": gates,
        "qualified": all(gates.values()),
        "limitations": [
            "synthetic known-cluster mechanism test only",
            "does not establish benefit inside RIPII",
            "does not compare downstream task performance",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--coarse-codes", type=int, default=4)
    parser.add_argument("--fine-codes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_qualification(
        seed=args.seed,
        steps=args.steps,
        samples=args.samples,
        code_dim=args.code_dim,
        coarse_codes=args.coarse_codes,
        fine_codes=args.fine_codes,
        learning_rate=args.learning_rate,
    )
    result["source_sha256"] = {
        "scripts/qualify_quantizer.py": _sha256(Path(__file__)),
        "ripii/models/quantizer.py": _sha256(ROOT / "ripii/models/quantizer.py"),
    }
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
