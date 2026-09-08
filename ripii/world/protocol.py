from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import PurePosixPath


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _validate_seeds(value, minimum: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) < minimum
        or len(set(value)) != len(value)
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in value
        )
    ):
        raise ValueError(f"protocol requires at least {minimum} unique nonnegative seeds")


def _validate_source_hashes(value) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("protocol requires source hashes")
    for path, digest in value.items():
        relative = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or relative is None
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("protocol contains an invalid source hash")


def _validate_experiment(value) -> None:
    if not isinstance(value, dict):
        raise ValueError("protocol requires an experiment object")
    required_positive = {
        "steps",
        "train_scenes",
        "eval_scenes",
        "train_horizon",
        "test_horizon",
        "rollout_steps",
        "batch_size",
        "hidden",
        "max_objects",
        "validate_every",
    }
    if any(
        not isinstance(value.get(key), int)
        or isinstance(value.get(key), bool)
        or value[key] < 1
        for key in required_positive
    ):
        raise ValueError("protocol experiment has invalid positive integer fields")
    for key in ("lr", "quantizer_weight", "global_coupling"):
        item = value.get(key)
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
        ):
            raise ValueError(f"protocol experiment has invalid {key}")
    if not 0 < value["lr"] < 1 or min(
        value["quantizer_weight"], value["global_coupling"]
    ) < 0:
        raise ValueError("protocol experiment has out-of-range numeric fields")
    if not isinstance(value.get("data_seed"), int) or isinstance(
        value.get("data_seed"), bool
    ):
        raise ValueError("protocol experiment requires an integer data seed")


def _validate_world(payload: dict) -> None:
    _validate_seeds(payload.get("seeds"), 1)
    variants = payload.get("variants")
    if (
        not isinstance(variants, list)
        or not variants
        or len(set(variants)) != len(variants)
        or any(
            item not in {"mlp", "graph", "transformer", "global_pool", "multiscale"}
            for item in variants
        )
    ):
        raise ValueError("world protocol has invalid model variants")
    bottlenecks = payload.get("bottlenecks")
    if (
        not isinstance(bottlenecks, list)
        or not bottlenecks
        or len(set(bottlenecks)) != len(bottlenecks)
        or any(item not in {"continuous", "fsq", "vq"} for item in bottlenecks)
    ):
        raise ValueError("world protocol has invalid bottlenecks")
    _validate_experiment(payload.get("experiment"))
    _validate_source_hashes(payload.get("source_sha256"))
    datasets = payload.get("datasets")
    expected_splits = {
        "train",
        "validation",
        "test",
        "more_objects",
        "composition",
        "fast",
    }
    if not isinstance(datasets, dict) or set(datasets) != expected_splits:
        raise ValueError("world protocol requires every declared dataset split")
    for split, spec in datasets.items():
        if (
            not isinstance(spec, dict)
            or spec.get("format") != "ripii-dataset-spec-v1"
            or spec.get("split") != split
            or not isinstance(spec.get("license"), str)
            or not spec["license"]
        ):
            raise ValueError("world protocol contains an invalid dataset specification")
    for key in ("selection", "training", "advancement_rule", "test_policy"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"world protocol is missing {key}")


def _validate_coupling(payload: dict) -> None:
    _validate_seeds(payload.get("seeds"), 2)
    if payload.get("models") != ["graph", "global_pool", "multiscale"]:
        raise ValueError("coupling protocol requires graph/global-pool/multiscale controls")
    if payload.get("bottleneck") != "continuous":
        raise ValueError("coupling protocol requires the continuous bottleneck")
    regimes = payload.get("regimes")
    if (
        not isinstance(regimes, dict)
        or set(regimes) != {"local", "coupled"}
        or regimes["local"] != 0
        or not isinstance(regimes["coupled"], (int, float))
        or isinstance(regimes["coupled"], bool)
        or not math.isfinite(regimes["coupled"])
        or regimes["coupled"] <= 0
    ):
        raise ValueError("coupling protocol has invalid regimes")
    _validate_experiment(payload.get("experiment_without_coupling"))
    _validate_source_hashes(payload.get("source_sha256"))
    for key in ("hypothesis", "h0", "h1", "decision_rule"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"coupling protocol is missing {key}")


@dataclass(frozen=True)
class ExperimentProtocol:
    """Validated protocol stored as canonical immutable JSON."""

    _canonical_json: str

    @classmethod
    def from_mapping(cls, payload: dict) -> ExperimentProtocol:
        try:
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            copied = json.loads(canonical, parse_constant=_reject_constant)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("protocol must be finite JSON") from exc
        if not isinstance(copied, dict):
            raise ValueError("protocol must be a JSON object")
        kind = copied.get("format")
        if kind == "ripii-world-protocol-v1":
            _validate_world(copied)
        elif kind == "ripii-coupling-protocol-v1":
            _validate_coupling(copied)
        else:
            raise ValueError("unsupported experiment protocol format")
        return cls(canonical)

    def as_dict(self) -> dict:
        return json.loads(self._canonical_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._canonical_json.encode()).hexdigest()
