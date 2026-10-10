from __future__ import annotations

import json
import os
import tempfile
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
        *,
        sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Inspect semantic dtypes before conversion: NaN/nonzero numeric masks
        # must not silently become live objects, nor complex states lose data.
        state = torch.as_tensor(state, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        if not state.is_floating_point() or not action.is_floating_point():
            raise TypeError("state and actions must be floating-point tensors")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        objects = self.model.max_objects
        if state.ndim != 3 or state.shape[0] == 0 or state.shape[1:] != (objects, 6):
            raise ValueError(f"state must have nonempty shape [batch, {objects}, 6]")
        if mask.shape != state.shape[:2]:
            raise ValueError("mask shape must match the state batch and objects")
        if sequence:
            if (
                action.ndim != 4
                or action.shape[0] != state.shape[0]
                or action.shape[2:] != (objects, 2)
            ):
                raise ValueError("actions must have shape [batch, time, objects, 2]")
        elif action.shape != (*state.shape[:2], 2):
            raise ValueError("action must have shape [batch, objects, 2]")
        # Validate the actual model dtype, including overflow on a float64 cast.
        state = state.to(dtype=torch.float32)
        action = action.to(dtype=torch.float32)
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise FloatingPointError("non-finite inference input")
        if not mask.any(dim=1).all():
            raise ValueError("every scene must contain at least one live object")
        if (state[..., 4:][mask] <= 0).any():
            raise ValueError("live objects require positive radius and mass")
        if (state.masked_select(~mask.unsqueeze(-1)) != 0).any():
            raise ValueError("padded state entries must be zero")
        inactive_actions = ~mask[:, None, :, None] if sequence else ~mask.unsqueeze(-1)
        if (action.masked_select(inactive_actions) != 0).any():
            raise ValueError("padded action entries must be zero")
        return state, action, mask

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
        state, actions, mask = self._inputs(state, actions, mask, sequence=True)
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
        payload = np.load(path, allow_pickle=False)
        if not isinstance(payload, np.lib.npyio.NpzFile):
            raise ValueError("inference input must be an NPZ archive")
        with payload:
            keys = set(payload.files)
            if len(keys) != len(payload.files):
                raise ValueError("NPZ array names must be unique")
            if keys not in ({"state", "action", "mask"}, {"state", "actions", "mask"}):
                raise ValueError(
                    "NPZ must contain state, mask, and exactly one of action/actions"
                )
            return {key: torch.from_numpy(payload[key].copy()) for key in keys}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid inference NPZ: {path}") from exc


def save_prediction_npz(path: str | Path, states: torch.Tensor) -> None:
    """Publish one complete prediction archive without overwriting any path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing prediction output: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, states=states.detach().cpu().numpy())
            handle.flush()
            os.fsync(handle.fileno())
        # Linking a complete same-directory file is an atomic, exclusive publish.
        # A concurrent creator (including a symlink) wins without being replaced.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def inspect_json(predictor: WorldPredictor) -> str:
    return json.dumps(predictor.inspect(), indent=2, allow_nan=False)
