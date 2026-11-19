"""Small, fail-closed execution contracts; no numerical imports or providers."""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

DEFAULT_SESSION_SECONDS = {
    "PREFLIGHT": 1800,
    "DEVELOPMENT_FULL": 24000,
    "RESUME": 1800,
    "ANALYZE_ONLY": 1800,
}

QUALIFICATION_HORIZONS = (1, 2, 4, 8, 16, 32, 64)


def validate_reference_result(name, result):
    """Require the complete frozen-reference comparison before admission.

    This validates a numerical child's retained receipt; it does not perform or
    substitute for that child's tensor comparisons.
    """
    if not isinstance(result, dict):
        raise ValueError("Frozen-reference result must be an object")
    for key, expected in {
        "model": name,
        "status": "PASS",
        "checkpoint_replay": "PASS",
        "invalid_input": "REJECTED",
    }.items():
        if result.get(key) != expected:
            raise ValueError("Frozen-reference result missing: " + key)
    checks = result.get("checks")
    if not isinstance(checks, list) or len(checks) != len(QUALIFICATION_HORIZONS):
        raise ValueError("Frozen-reference horizon coverage differs")
    for row, horizon in zip(checks, QUALIFICATION_HORIZONS, strict=True):
        if (
            not isinstance(row, dict)
            or type(row.get("horizon")) is not int
            or row["horizon"] != horizon
            or type(row.get("active_gradients")) is not int
            or row["active_gradients"] < 1
        ):
            raise ValueError("Frozen-reference horizon/gradient coverage differs")
        for key in ("prediction_max_abs", "gradient_max_abs"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid frozen-reference measurement: " + key)
    value = result.get("optimizer_max_abs")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid frozen-reference optimizer measurement")


def worker_environment(parent=None):
    """Preserve the frozen launcher contract before any child CUDA import."""
    return dict(
        os.environ if parent is None else parent,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or ".." in p.parts or str(p) != name:
        raise ValueError("Unsafe artifact path")
    return p


def reservation(prior, seconds, reserve=900, cap=64800, devices=2):
    values = (prior, seconds, reserve, cap, devices)
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in values
    ):
        raise ValueError("Finite numeric budget required")
    if (
        prior < 0
        or seconds <= 0
        or reserve < 0
        or cap <= 0
        or devices < 1
        or int(devices) != devices
    ):
        raise ValueError("Invalid budget")
    charge = devices * (seconds + reserve)
    if prior + charge > cap:
        raise ValueError("Cumulative allocation exceeds frozen cap")
    return {
        "prior": prior,
        "reserved_gpu_seconds": charge,
        "cumulative_ceiling_gpu_seconds": prior + charge,
        "cap": cap,
    }


def verify_inventory(root, files):
    root = Path(root)
    for name, expected in files.items():
        path = root.joinpath(*safe_relative(name).parts)
        if (
            not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or not path.is_file()
            or digest(path) != expected
        ):
            raise ValueError("Artifact identity differs: " + name)


def session_reservation(prior, seconds, mode):
    """Protect one 30-minute continuation before admitting full development.

    Future recovery headroom is reserved for planning, not charged twice. The
    ordinary cumulative ceiling remains the charge carried into a later session.
    """
    current = reservation(prior, seconds)
    if mode == "DEVELOPMENT_FULL":
        continuation = reservation(
            current["cumulative_ceiling_gpu_seconds"],
            DEFAULT_SESSION_SECONDS["RESUME"],
        )
        current.update(
            protected_resume_seconds=DEFAULT_SESSION_SECONDS["RESUME"],
            protected_resume_gpu_seconds=continuation["reserved_gpu_seconds"],
            cumulative_with_reserved_resume_gpu_seconds=continuation[
                "cumulative_ceiling_gpu_seconds"
            ],
        )
    return current


def admit_full(source_digest, receipt):
    required = {
        "status": "PASS",
        "source_sha256": source_digest,
        "all_models": 5,
        "curriculum_horizons": 7,
        "resume_new_process": "PASS",
        "gate_intervention": "PASS",
        "capture_eager": "PASS",
        "capture_lr_change": "PASS",
        "frozen_reference_equivalence": "PASS",
        "test_data_opened": False,
    }
    for key, expected in required.items():
        if type(receipt.get(key)) is not type(expected) or receipt.get(key) != expected:
            raise ValueError("Full execution qualification missing: " + key)


def identity(files):
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
