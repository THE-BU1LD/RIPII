from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ripii.world.experiment import atomic_save
from ripii.world.inference import (
    WorldPredictor,
    load_inference_npz,
    save_prediction_npz,
)
from ripii.world.models import WorldModel
from ripii.world.physics import sample_scene


def checkpoint(path: Path) -> Path:
    model = WorldModel("graph", hidden=16, max_objects=8)
    atomic_save(
        path,
        {
            "format": "ripii-world-v1",
            "model": model.state_dict(),
            "model_spec": model.spec,
            "completed_steps": 1,
            "best_validation": 0.5,
        },
    )
    return path


def test_supported_predict_and_rollout_api(tmp_path: Path):
    predictor = WorldPredictor.from_checkpoint(checkpoint(tmp_path / "model.pt"))
    state, mask = sample_scene(torch.Generator().manual_seed(4), 3)
    action = torch.zeros(1, 8, 2)
    predicted = predictor.predict(state[None], action, mask[None])
    trajectory = predictor.rollout(
        state[None], torch.zeros(1, 3, 8, 2), mask[None]
    )
    assert predicted.shape == (1, 8, 6)
    assert trajectory.shape == (1, 4, 8, 6)
    assert predictor.inspect()["parameters"] > 0


def test_npz_io_is_non_pickled_and_contract_checked(tmp_path: Path):
    source = tmp_path / "input.npz"
    np.savez(
        source,
        state=np.zeros((1, 8, 6), dtype=np.float32),
        action=np.zeros((1, 8, 2), dtype=np.float32),
        mask=np.ones((1, 8), dtype=np.bool_),
    )
    payload = load_inference_npz(source)
    assert set(payload) == {"state", "action", "mask"}
    output = tmp_path / "prediction.npz"
    save_prediction_npz(output, torch.zeros(1, 8, 6))
    with np.load(output, allow_pickle=False) as result:
        assert result["states"].shape == (1, 8, 6)
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, unexpected=np.zeros(1))
    with pytest.raises(ValueError, match="invalid inference NPZ"):
        load_inference_npz(invalid)
