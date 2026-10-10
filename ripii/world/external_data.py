"""Manifest-verified interchange format for object-state trajectory datasets."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .data import validate_tensor_dataset

FORMAT = "ripii-trajectory-dataset-v1"
GROUPED_FORMAT = "ripii-trajectory-dataset-v2"
REQUIRED_SPLITS = ("train", "validation", "test")
TRAJECTORY_FINGERPRINT = "ripii-trajectory-content-v1"
SOURCE_GROUP_SCHEMA = "ripii-source-episode-v1"


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"external dataset manifest has a duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_source_groups(manifest: dict) -> None:
    """Admit declared grouping without opening any trajectory archive.

    Identities are exact caller declarations, not proof of physical independence.
    A source/episode pair may have several windows, but only in one split.
    """
    groups = manifest.get("source_groups")
    if not isinstance(groups, dict) or set(groups) != set(manifest["artifacts"]):
        raise ValueError("source_groups must cover exactly all declared splits")
    row_ids = set()
    group_split = {}
    for split, rows in groups.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"source_groups must contain nonempty row lists: {split}")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"trajectory_id", "source_id", "episode_id"}:
                raise ValueError(f"invalid source_groups row schema: {split}")
            row_id = row["trajectory_id"]
            if isinstance(row_id, bool) or not isinstance(row_id, int) or not -(2**63) <= row_id < 2**63:
                raise ValueError("source_groups trajectory_id must be a signed int64 integer")
            if row_id in row_ids:
                raise ValueError("source_groups contains a duplicate trajectory_id")
            row_ids.add(row_id)
            pair = (row["source_id"], row["episode_id"])
            if any(not isinstance(part, str) or not 1 <= len(part) <= 256 or part.strip() != part or not part.isprintable() for part in pair):
                raise ValueError("source_groups source_id and episode_id must be canonical printable strings of 1 to 256 characters")
            if pair in group_split and group_split[pair] != split:
                raise ValueError(f"declared source episode overlaps between {group_split[pair]} and {split}")
            group_split[pair] = split


def _source_group_record(manifest: dict, split: str, row_ids: list[int]) -> dict:
    rows = {row["trajectory_id"]: row for row in manifest["source_groups"][split]}
    if set(row_ids) != set(rows):
        raise ValueError(f"source_groups trajectory IDs do not exactly match loaded rows: {split}")
    records = []
    for row_id in row_ids:
        row = rows[row_id]
        identity = {"schema": SOURCE_GROUP_SCHEMA, "source_id": row["source_id"], "episode_id": row["episode_id"]}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        records.append({**row, "group_sha256": digest})
    unique = sorted({row["group_sha256"] for row in records})
    return {
        "schema": SOURCE_GROUP_SCHEMA,
        "unit": "declared_source_episode",
        "unique_groups": len(unique),
        "cohort_sha256": hashlib.sha256("".join(unique).encode("ascii")).hexdigest(),
        "records": records,
        "limitation": "Caller-declared identity; does not authenticate provenance, aliases, or statistical independence.",
    }


def _manifest(root: Path) -> tuple[dict, str]:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("external dataset manifest is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("external dataset manifest is invalid JSON") from exc
    required_strings = (
        "dataset_id",
        "version",
        "license",
        "units",
        "preprocessing",
        "split_policy",
    )
    if (
        not isinstance(value, dict)
        or value.get("format") not in (FORMAT, GROUPED_FORMAT)
        or any(
            not isinstance(value.get(key), str) or not value[key].strip()
            for key in required_strings
        )
        or not isinstance(value.get("artifacts"), dict)
        or not isinstance(value.get("observation_dt"), (int, float))
        or isinstance(value.get("observation_dt"), bool)
        or not np.isfinite(value["observation_dt"])
        or value["observation_dt"] <= 0
        or not set(REQUIRED_SPLITS) <= set(value["artifacts"])
    ):
        raise ValueError("external dataset manifest violates the required schema")
    if value["format"] == GROUPED_FORMAT:
        _validate_source_groups(value)
    return value, hashlib.sha256(raw).hexdigest()


def _artifact_bytes(root: Path, entry: dict, split: str) -> bytes:
    relative = entry.get("path") if isinstance(entry, dict) else None
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    path = root / relative if isinstance(relative, str) else root
    # Check every lexical component before resolution: a safe-looking leaf
    # below a symlinked directory can otherwise leave the declared dataset root.
    unsafe_component = False
    if pure is not None and not pure.is_absolute():
        cursor = root
        for component in pure.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                unsafe_component = True
                break
    try:
        contained = path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        contained = False
    if (
        pure is None
        or pure.is_absolute()
        or ".." in pure.parts
        or unsafe_component
        or not contained
        or path.is_symlink()
        or not path.is_file()
        or not isinstance(entry.get("bytes"), int)
        or isinstance(entry.get("bytes"), bool)
        or entry["bytes"] < 1
        or not isinstance(entry.get("sha256"), str)
    ):
        raise ValueError(f"external dataset artifact failed verification: {split}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"external dataset artifact failed verification: {split}") from exc
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"external dataset artifact failed verification: {split}")
    return raw


def load_trajectory_split(
    root: str | Path, split: str, *, expected_manifest_sha256: str | None = None
) -> tuple[dict[str, torch.Tensor], dict]:
    if expected_manifest_sha256 is not None and (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_manifest_sha256)
    ):
        raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256 digest")
    root = Path(root).resolve()
    manifest, manifest_sha256 = _manifest(root)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("external dataset manifest changed since verification")
    return _load_trajectory_split(root, split, manifest, manifest_sha256)


def _load_trajectory_split(
    root: Path, split: str, manifest: dict, manifest_sha256: str
) -> tuple[dict[str, torch.Tensor], dict]:
    if split not in manifest["artifacts"]:
        raise ValueError(f"external dataset does not declare split: {split}")
    entry = manifest["artifacts"][split]
    raw = _artifact_bytes(root, entry, split)
    try:
        # Parse exactly the bytes whose length and digest were admitted. A
        # second path open could see a concurrent replacement after hashing.
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) != 4 or set(archive.files) != {"states", "actions", "mask", "ids"}:
                raise ValueError("trajectory NPZ contains unexpected arrays")
            data = {
                "states": torch.from_numpy(archive["states"].copy()),
                "actions": torch.from_numpy(archive["actions"].copy()),
                "mask": torch.from_numpy(archive["mask"].copy()),
                "ids": torch.from_numpy(archive["ids"].copy()),
            }
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"external trajectory split is invalid: {split}") from exc
    states, actions = data["states"], data["actions"]
    if states.ndim != 4 or actions.ndim != 4:
        raise ValueError("external trajectory arrays have invalid ranks")
    scenes, observations, max_objects, features = states.shape
    if features != 6 or actions.shape != (scenes, observations - 1, max_objects, 2):
        raise ValueError("external trajectory state/action dimensions do not align")
    validate_tensor_dataset(scenes, observations - 1, max_objects, data)
    record = {
        "format": manifest["format"],
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "license": manifest["license"],
        "units": manifest["units"],
        "preprocessing": manifest["preprocessing"],
        "split_policy": manifest["split_policy"],
        "observation_dt": float(manifest["observation_dt"]),
        "split": split,
        "scenes": scenes,
        "horizon": observations - 1,
        "max_objects": max_objects,
        "source_path": entry["path"],
        "source_sha256": entry["sha256"],
        "manifest_sha256": manifest_sha256,
    }
    if manifest["format"] == GROUPED_FORMAT:
        record["source_groups"] = _source_group_record(manifest, split, data["ids"].tolist())
    return data, record


def _trajectory_content_sha256(data: dict[str, torch.Tensor], scene: int) -> str:
    """Hash one validated trajectory's values independently of its assigned ID.

    Zero padding, signed zero, storage layout, and floating-point dtype do not
    distinguish otherwise identical observations. Object and time order do.
    This detects exact copies, not overlapping windows or common source scenes.
    """
    digest = hashlib.sha256(TRAJECTORY_FINGERPRINT.encode("ascii"))
    live = data["mask"][scene].nonzero(as_tuple=False).flatten()
    for key in ("states", "actions"):
        values = (
            data[key][scene]
            .index_select(1, live)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        canonical = np.array(values, dtype="<f8", order="C", copy=True)
        canonical[canonical == 0] = 0.0
        header = json.dumps(
            {"field": key, "shape": list(canonical.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def verify_trajectory_dataset(root: str | Path) -> dict:
    root = Path(root).resolve()
    manifest, manifest_sha256 = _manifest(root)
    records = {}
    ids_by_split = {}
    content_by_split = {}
    for split in manifest["artifacts"]:
        data, record = _load_trajectory_split(root, split, manifest, manifest_sha256)
        records[split] = record
        ids_by_split[split] = set(data["ids"].tolist())
        content = {
            _trajectory_content_sha256(data, scene)
            for scene in range(record["scenes"])
        }
        content_by_split[split] = content
        record["content_verification"] = {
            "fingerprint_schema": TRAJECTORY_FINGERPRINT,
            "unique_trajectories": len(content),
            "cohort_sha256": hashlib.sha256(
                "".join(sorted(content)).encode("ascii")
            ).hexdigest(),
        }
    for index, left in enumerate(ids_by_split):
        for right in list(ids_by_split)[index + 1 :]:
            if ids_by_split[left] & ids_by_split[right]:
                raise ValueError(f"trajectory IDs overlap between {left} and {right}")
            if content_by_split[left] & content_by_split[right]:
                raise ValueError(
                    f"trajectory contents overlap between {left} and {right}; "
                    "changing IDs does not create independent trajectories"
                )
    result = {
        "status": "PASS",
        "format": manifest["format"],
        "dataset_id": manifest["dataset_id"],
        "version": manifest["version"],
        "manifest_sha256": manifest_sha256,
        "splits": records,
        "content_overlap_check": {
            "fingerprint_schema": TRAJECTORY_FINGERPRINT,
            "status": "PASS",
            "scope": "exact ordered state/action trajectories after removing padding",
            "limitation": (
                "Does not establish source independence or detect overlapping windows, "
                "approximate duplicates, or reordered live objects."
            ),
        },
    }
    if manifest["format"] == GROUPED_FORMAT:
        result["source_group_overlap_check"] = {
            "schema": SOURCE_GROUP_SCHEMA,
            "status": "PASS",
            "scope": "Exact caller-declared source/episode pairs occur in one split; every loaded row has one declaration.",
            "limitation": "Distinct declared identities do not prove independent sources or exclude undeclared aliases.",
        }
    return result
