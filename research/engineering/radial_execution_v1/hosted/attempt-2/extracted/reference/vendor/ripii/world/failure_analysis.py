from __future__ import annotations

import math

import torch


def transition_regime_masks(
    states: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    near_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Classify object transitions using only the pre-transition true state.

    Categories intentionally overlap except for ``free_flight``. A contact object can
    also be forced or at a wall; counts are therefore reported with every metric.
    """
    if (
        states.ndim != 4
        or states.shape[-1] != 6
        or actions.shape != (*states.shape[:2], states.shape[2], 2)
        or mask.shape != (states.shape[0], states.shape[2])
        or states.shape[1] < 1
    ):
        raise ValueError("invalid trajectory, action, or mask shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if not math.isfinite(near_margin) or near_margin < 0:
        raise ValueError("near_margin must be finite and non-negative")
    if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise FloatingPointError("non-finite regime input")
    property_mask = mask[:, None, :, None].expand_as(states[..., 4:])
    if not mask.any(dim=1).all() or (states[..., 4:][property_mask] <= 0).any():
        raise ValueError("every scene needs live objects with positive radius and mass")

    position, radius = states[..., :2], states[..., 4]
    n = states.shape[2]
    relative = position.unsqueeze(3) - position.unsqueeze(2)
    distance = relative.norm(dim=-1)
    live_pairs = mask[:, None, :, None] & mask[:, None, None, :]
    live_pairs = live_pairs & ~torch.eye(n, dtype=torch.bool, device=states.device)
    threshold = radius.unsqueeze(3) + radius.unsqueeze(2)
    contact = ((distance <= threshold) & live_pairs).any(dim=3)
    near = ((distance <= threshold + near_margin) & live_pairs).any(dim=3) & ~contact
    wall = (position.abs() >= (1.0 - radius).unsqueeze(-1)).any(dim=-1)
    forced = actions.norm(dim=-1) > 0
    live = mask[:, None, :].expand_as(contact)
    return {
        "contact": contact & live,
        "near_contact": near & live,
        "wall": wall & live,
        "forced": forced & live,
        "free_flight": live & ~contact & ~near & ~wall & ~forced,
        "all": live,
    }


@torch.no_grad()
def evaluate_failure_regimes(
    model,
    data: dict[str, torch.Tensor],
    *,
    near_margin: float = 0.05,
) -> dict[str, dict[str, float | int]]:
    required = {"states", "actions", "mask"}
    if not required <= data.keys():
        raise ValueError(f"data is missing keys: {sorted(required - data.keys())}")
    states, actions, mask = data["states"], data["actions"], data["mask"]
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError("trajectory must contain one more state than action")
    model.eval()
    current = states[:, 0]
    predictions = []
    for step in range(actions.shape[1]):
        current = model(current, actions[:, step], mask)
        predictions.append(current)
    predicted = torch.stack(predictions, dim=1)
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("non-finite failure-analysis rollout")
    error = predicted[..., :4] - states[:, 1:, :, :4]
    regimes = transition_regime_masks(
        states[:, :-1], actions, mask, near_margin=near_margin
    )
    result = {}
    for name, regime in regimes.items():
        count = int(regime.sum())
        entry: dict[str, float | int] = {"object_transitions": count}
        if count:
            entry["position_rmse"] = float(
                (error[..., :2].square()[regime].sum() / (count * 2)).sqrt()
            )
            entry["velocity_rmse"] = float(
                (error[..., 2:4].square()[regime].sum() / (count * 2)).sqrt()
            )
        result[name] = entry
    return result
