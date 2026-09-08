from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ripii.world.data import DatasetSpec
from ripii.world.experiment import Experiment
from ripii.world.protocol import ExperimentProtocol


def _world_protocol() -> dict:
    datasets = {
        split: DatasetSpec(split, 2, 2, 7, 8).as_record()
        for split in (
            "train",
            "validation",
            "test",
            "more_objects",
            "composition",
            "fast",
        )
    }
    return {
        "format": "ripii-world-protocol-v1",
        "seeds": [3],
        "variants": ["graph", "multiscale"],
        "bottlenecks": ["continuous"],
        "experiment": Experiment().__dict__,
        "source_sha256": {"source.py": "a" * 64},
        "datasets": datasets,
        "selection": "validation only",
        "training": "paired",
        "advancement_rule": "predeclared",
        "test_policy": "held out",
    }


def test_protocol_is_validated_immutable_and_hashable() -> None:
    payload = _world_protocol()
    protocol = ExperimentProtocol.from_mapping(payload)
    payload["seeds"].append(5)
    assert protocol.as_dict()["seeds"] == [3]
    assert len(protocol.sha256) == 64
    with pytest.raises(FrozenInstanceError):
        protocol._canonical_json = "changed"


def test_protocol_rejects_incomplete_grids_and_nonfinite_values() -> None:
    payload = _world_protocol()
    payload["seeds"] = [3, 3]
    with pytest.raises(ValueError, match="unique"):
        ExperimentProtocol.from_mapping(payload)
    payload = _world_protocol()
    payload["experiment"]["lr"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        ExperimentProtocol.from_mapping(payload)
    payload = _world_protocol()
    del payload["datasets"]["fast"]
    with pytest.raises(ValueError, match="every declared"):
        ExperimentProtocol.from_mapping(payload)
