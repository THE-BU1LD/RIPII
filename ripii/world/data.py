from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from .physics import SPLITS, Physics, make_dataset


@dataclass(frozen=True)
class DatasetAdapter:
    dataset_id: str
    versions: tuple[str, ...]
    license_id: str
    supported_splits: tuple[str, ...]
    feature_contract: dict
    split_policy: str
    preprocessing: str
    loader: Callable[[DatasetSpec], dict[str, torch.Tensor]]


_ADAPTERS: dict[str, DatasetAdapter] = {}


def register_dataset_adapter(adapter: DatasetAdapter) -> None:
    if (
        not isinstance(adapter.dataset_id, str)
        or not adapter.dataset_id
        or not adapter.versions
        or len(set(adapter.versions)) != len(adapter.versions)
        or any(not isinstance(version, str) or not version for version in adapter.versions)
        or not isinstance(adapter.license_id, str)
        or not adapter.license_id
        or not adapter.supported_splits
        or len(set(adapter.supported_splits)) != len(adapter.supported_splits)
        or any(not isinstance(split, str) or not split for split in adapter.supported_splits)
        or not isinstance(adapter.feature_contract, dict)
        or not {"state", "action", "mask"} <= adapter.feature_contract.keys()
        or not isinstance(adapter.split_policy, str)
        or not adapter.split_policy
        or not isinstance(adapter.preprocessing, str)
        or not adapter.preprocessing
        or not callable(adapter.loader)
        or adapter.dataset_id in _ADAPTERS
    ):
        raise ValueError("invalid or duplicate dataset adapter")
    _ADAPTERS[adapter.dataset_id] = adapter


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable identity and generation contract for one trajectory split."""

    split: str
    scenes: int
    horizon: int
    seed: int
    max_objects: int
    physics: Physics = Physics()
    dataset_id: str = "ripii.synthetic.soft_discs"
    version: str = "1"

    def validate(self) -> None:
        adapter = _ADAPTERS.get(self.dataset_id)
        if (
            adapter is None
            or self.version not in adapter.versions
            or self.split not in adapter.supported_splits
            or not isinstance(self.scenes, int)
            or not isinstance(self.horizon, int)
            or not isinstance(self.seed, int)
            or not isinstance(self.max_objects, int)
            or isinstance(self.scenes, bool)
            or isinstance(self.horizon, bool)
            or isinstance(self.seed, bool)
            or isinstance(self.max_objects, bool)
            or self.scenes < 1
            or self.horizon < 1
            or not 5 <= self.max_objects <= 16
            or not isinstance(self.physics, Physics)
        ):
            raise ValueError("invalid synthetic dataset specification")

    def as_record(self) -> dict:
        self.validate()
        adapter = _ADAPTERS[self.dataset_id]
        return {
            "format": "ripii-dataset-spec-v1",
            "dataset_id": self.dataset_id,
            "version": self.version,
            "license": adapter.license_id,
            "split": self.split,
            "scenes": self.scenes,
            "horizon": self.horizon,
            "seed": self.seed,
            "max_objects": self.max_objects,
            "physics": asdict(self.physics),
            "feature_contract": json.loads(json.dumps(adapter.feature_contract)),
            "split_policy": adapter.split_policy,
            "preprocessing": adapter.preprocessing,
            "license_boundary": (
                "Adapter metadata is provenance, not a legal determination; verify "
                "redistribution terms before retaining external data"
            ),
        }


def _tensor_sha256(data: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(data):
        value = data[key].detach().cpu().contiguous()
        header = json.dumps(
            {"key": key, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_dataset(spec: DatasetSpec, data: dict[str, torch.Tensor]) -> None:
    spec.validate()
    if set(data) != {"states", "actions", "mask", "ids"} or not all(
        isinstance(value, torch.Tensor) for value in data.values()
    ):
        raise ValueError("dataset must contain exactly states, actions, mask, and ids")
    expected_state = (spec.scenes, spec.horizon + 1, spec.max_objects, 6)
    expected_action = (spec.scenes, spec.horizon, spec.max_objects, 2)
    if (
        data["states"].shape != expected_state
        or data["actions"].shape != expected_action
        or data["mask"].shape != (spec.scenes, spec.max_objects)
        or data["ids"].shape != (spec.scenes,)
        or not data["states"].is_floating_point()
        or not data["actions"].is_floating_point()
        or data["mask"].dtype != torch.bool
        or data["ids"].dtype != torch.int64
    ):
        raise ValueError("dataset tensors violate the declared shape or dtype contract")
    if not torch.isfinite(data["states"]).all() or not torch.isfinite(
        data["actions"]
    ).all():
        raise FloatingPointError("dataset contains non-finite states or actions")
    if not data["mask"].any(dim=1).all() or data["ids"].unique().numel() != spec.scenes:
        raise ValueError("dataset requires live objects and unique trajectory IDs")
    padded = ~data["mask"]
    if data["states"].masked_select(padded[:, None, :, None]).ne(0).any() or data[
        "actions"
    ].masked_select(padded[:, None, :, None]).ne(0).any():
        raise ValueError("padded states and actions must be zero")
    live_properties = data["states"][..., 4:][
        data["mask"][:, None, :, None].expand_as(data["states"][..., 4:])
    ]
    if live_properties.numel() == 0 or not torch.isfinite(live_properties).all():
        raise ValueError("live object properties are missing or invalid")
    properties = data["states"][..., 4:]
    if (properties[data["mask"][:, None, :].expand_as(properties[..., 0])] <= 0).any():
        raise ValueError("live radius and mass values must be positive")


def load_dataset(spec: DatasetSpec) -> tuple[dict[str, torch.Tensor], dict]:
    spec.validate()
    data = _ADAPTERS[spec.dataset_id].loader(spec)
    validate_dataset(spec, data)
    record = spec.as_record()
    record["content_sha256"] = _tensor_sha256(data)
    record["tensor_bytes"] = sum(
        value.numel() * value.element_size() for value in data.values()
    )
    if not math.isfinite(record["tensor_bytes"]):
        raise FloatingPointError("invalid dataset byte count")
    return data, record


def _load_synthetic(spec: DatasetSpec) -> dict[str, torch.Tensor]:
    return make_dataset(
        spec.split,
        spec.scenes,
        spec.horizon,
        spec.seed,
        spec.max_objects,
        spec.physics,
    )


register_dataset_adapter(
    DatasetAdapter(
        dataset_id="ripii.synthetic.soft_discs",
        versions=("1",),
        license_id="NOASSERTION",
        supported_splits=SPLITS,
        feature_contract={
            "state": ["x", "y", "vx", "vy", "radius", "mass"],
            "state_units": [
                "arena_unit",
                "arena_unit",
                "arena_unit/time",
                "arena_unit/time",
                "arena_unit",
                "mass_unit",
            ],
            "action": ["force_x", "force_y"],
            "action_units": ["mass*arena_unit/time^2"] * 2,
            "mask": "true for live objects; padded states/actions are zero",
        },
        split_policy=(
            "deterministic disjoint RNG domain per named split; more_objects, "
            "composition, and fast alter one declared factor"
        ),
        preprocessing="none; generated object states are consumed directly",
        loader=_load_synthetic,
    )
)
