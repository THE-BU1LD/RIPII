from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ripii.world.data import (
    DatasetAdapter,
    DatasetSpec,
    load_dataset,
    register_dataset_adapter,
    validate_dataset,
)
from ripii.world.physics import SPLITS, Physics, make_dataset


def test_dataset_spec_is_deterministic_validated_and_fingerprinted() -> None:
    spec = DatasetSpec("train", 4, 3, 17, 8, Physics(global_coupling=0.2))
    first, first_record = load_dataset(spec)
    second, second_record = load_dataset(spec)
    assert first_record == second_record
    assert len(first_record["content_sha256"]) == 64
    assert first_record["license"] == "NOASSERTION"
    assert torch.equal(first["states"], second["states"])
    assert load_dataset(replace(spec, split="test"))[1]["content_sha256"] != first_record[
        "content_sha256"
    ]


def test_dataset_contract_rejects_corruption() -> None:
    spec = DatasetSpec("train", 2, 2, 7, 8)
    data, _ = load_dataset(spec)
    data["states"][0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        validate_dataset(spec, data)
    data, _ = load_dataset(spec)
    padded = (~data["mask"][0]).nonzero()[0, 0]
    data["actions"][0, 0, padded, 0] = 1
    with pytest.raises(ValueError, match="padded"):
        validate_dataset(spec, data)
    with pytest.raises(ValueError):
        DatasetSpec("unknown", 2, 2, 7, 8).validate()


def test_dataset_registry_accepts_an_explicit_external_adapter() -> None:
    def loader(spec: DatasetSpec):
        return make_dataset(
            spec.split,
            spec.scenes,
            spec.horizon,
            spec.seed,
            spec.max_objects,
            spec.physics,
        )

    adapter = DatasetAdapter(
        dataset_id="test.external.fixture",
        versions=("2026-09",),
        license_id="CC0-1.0",
        supported_splits=SPLITS,
        feature_contract={
            "state": ["x", "y", "vx", "vy", "radius", "mass"],
            "action": ["force_x", "force_y"],
            "mask": "true for live objects",
        },
        split_policy="test-only adapter using declared split names",
        preprocessing="identity",
        loader=loader,
    )
    register_dataset_adapter(adapter)
    spec = DatasetSpec(
        "test", 2, 2, 7, 8, dataset_id=adapter.dataset_id, version="2026-09"
    )
    _, record = load_dataset(spec)
    assert record["license"] == "CC0-1.0"
    with pytest.raises(ValueError, match="duplicate"):
        register_dataset_adapter(adapter)
