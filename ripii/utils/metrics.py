
from __future__ import annotations

import torch


def sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((sanitize(x) - sanitize(y)) ** 2)


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(sanitize(x), dim=-1)
    y = torch.nn.functional.normalize(sanitize(y), dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - cosine_distance(x, y)


def mean_pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(sanitize(x), dim=-1)
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    sim = x @ x.T
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return sim.masked_select(mask).mean()


def variance_penalty(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = sanitize(x)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.relu(1.0 - std).mean()


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    x = sanitize(x)
    x = x - x.mean(dim=0, keepdim=True)
    return x.T @ x / max(1, x.shape[0] - 1)


def covariance_penalty(x: torch.Tensor) -> torch.Tensor:
    cov = covariance_matrix(x)
    d = torch.diag(cov)
    off = cov - torch.diag(d)
    return off.pow(2).mean()


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    probs = sanitize(probs).clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1).mean()


def perplexity_from_probs(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.exp(entropy_from_probs(probs, eps))


def spectral_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xf = torch.fft.rfft(sanitize(x), dim=-1)
    yf = torch.fft.rfft(sanitize(y), dim=-1)
    return torch.mean((xf.abs() - yf.abs()) ** 2)


def principal_angles(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = sanitize(u)
    v = sanitize(v)
    svals = torch.linalg.svdvals(u.T @ v).clamp(0.0, 1.0)
    return torch.arccos(svals)


def principal_angle_mean(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return principal_angles(u, v).mean()


def subspace_overlap(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = sanitize(u)
    v = sanitize(v)
    svals = torch.linalg.svdvals(u.T @ v).clamp(0.0, 1.0)
    return (svals ** 2).mean()


def effective_rank(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    cov = covariance_matrix(x)
    vals = torch.linalg.svdvals(cov).clamp_min(eps)
    p = vals / vals.sum().clamp_min(eps)
    ent = -(p * p.log()).sum()
    return torch.exp(ent)


def probe_accuracy(features: torch.Tensor, labels: torch.Tensor, num_classes: int, ridge: float = 1e-2) -> float:
    x = sanitize(features.detach()).double()
    y = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).double()
    xtx = x.T @ x
    eye = torch.eye(xtx.shape[0], dtype=x.dtype, device=x.device)
    w = torch.linalg.solve(xtx + ridge * eye, x.T @ y)
    preds = (x @ w).argmax(dim=-1)
    return float((preds == labels.long()).double().mean().item())


def ridge_probe_accuracy(features: torch.Tensor, labels: torch.Tensor, num_classes: int, ridge: float = 1e-2) -> float:
    return probe_accuracy(features, labels, num_classes, ridge)


def hutchinson_jacobian_norm(fn, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    x = sanitize(x).detach().requires_grad_(True)
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    for _ in range(num_samples):
        v = torch.randn_like(x)
        y = sanitize(fn(x))
        scalar = (y * v).sum()
        grad = torch.autograd.grad(scalar, x, retain_graph=True, create_graph=False)[0]
        total = total + grad.pow(2).sum(dim=-1).mean()
    return total / num_samples
