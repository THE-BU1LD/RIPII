from __future__ import annotations

import pytest

from ripii.world.data import DatasetSpec, load_dataset
from ripii.world.models import WorldModel
from ripii.world.profiling import ProfileConfig, profile_rollout


def test_profile_rollout_warms_up_repeats_and_reports_boundaries() -> None:
    data, _ = load_dataset(DatasetSpec("test", 2, 2, 7, 8))
    model = WorldModel("graph", hidden=8, max_objects=8)
    result = profile_rollout(model, data, ProfileConfig(1, 5, False))
    assert result["format"] == "ripii-world-profile-v1"
    assert result["measured_runs"] == 5
    assert result["scene_steps_per_second_at_mean"] > 0
    assert result["recognized_operator_flops_per_rollout"] is None
    assert "lower-bound proxy" in result["flop_boundary"]
    with pytest.raises(ValueError, match="profiling requires"):
        profile_rollout(model, data, ProfileConfig(0, 2, False))
