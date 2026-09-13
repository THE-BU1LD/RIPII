from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .experiment import load_model
from .models import rollout as rollout_model


class WorldPredictor:
    """Stable inference wrapper for a trained RIPII object-state model."""

    def __init__(self, model, checkpoint: dict[str, Any], device: str = "cpu"):
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = requested
        self.model = model.to(self.device).eval()
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu"
    ) -> WorldPredictor:
        model, checkpoint = load_model(path)
        return cls(model, checkpoint, device)

    def _inputs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(state, dtype=torch.float32, device=self.device),
            torch.as_tensor(action, dtype=torch.float32, device=self.device),
            torch.as_tensor(mask, dtype=torch.bool, device=self.device),
        )

    @torch.inference_mode()
    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        state, action, mask = self._inputs(state, action, mask)
        return self.model(state, action, mask)

    @torch.inference_mode()
    def rollout(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        state, actions, mask = self._inputs(state, actions, mask)
        return rollout_model(self.model, state, actions, mask)

    def inspect(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "format": self.checkpoint["format"],
            "model_spec": dict(self.checkpoint["model_spec"]),
            "completed_steps": self.checkpoint["completed_steps"],
            "best_validation": self.checkpoint["best_validation"],
            "parameters": parameters,
            "device": str(self.device),
            "datasets": self.checkpoint.get("datasets"),
        }


def load_inference_npz(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a non-pickled inference payload with an explicit tensor contract."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("inference input must be a regular non-symlink NPZ file")
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if keys not in ({"state", "action", "mask"}, {"state", "actions", "mask"}):
                raise ValueError(
                    "NPZ must contain state, mask, and exactly one of action/actions"
                )
            return {key: torch.from_numpy(payload[key].copy()) for key in keys}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid inference NPZ: {path}") from exc


def save_prediction_npz(path: str | Path, states: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, states=states.detach().cpu().numpy())
    temporary.replace(path)


def inspect_json(predictor: WorldPredictor) -> str:
    return json.dumps(predictor.inspect(), indent=2, allow_nan=False)
