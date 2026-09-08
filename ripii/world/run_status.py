from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_TRANSITIONS = {
    "planned": {"running", "failed"},
    "running": {"complete", "failed"},
    "complete": {"failed"},
    "failed": set(),
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_atomic(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@dataclass
class RunTracker:
    path: Path
    payload: dict

    @classmethod
    def create(
        cls, directory: Path, *, run_kind: str, protocol_sha256: str
    ) -> RunTracker:
        if not run_kind or len(protocol_sha256) != 64:
            raise ValueError("run tracker requires kind and protocol SHA-256")
        path = directory / "status.json"
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing run status: {path}")
        now = _timestamp()
        payload = {
            "format": "ripii-run-status-v1",
            "run_kind": run_kind,
            "protocol_sha256": protocol_sha256,
            "state": "planned",
            "events": [{"state": "planned", "at": now}],
        }
        # Exclusive creation prevents accidental reuse of a run directory.
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return cls(path, payload)

    @classmethod
    def open(cls, directory: Path) -> RunTracker:
        path = directory / "status.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("run status is missing or unsafe")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("run status is unreadable") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "ripii-run-status-v1"
            or payload.get("state") not in _TRANSITIONS
            or not isinstance(payload.get("events"), list)
            or not payload["events"]
        ):
            raise ValueError("invalid run status")
        return cls(path, payload)

    def transition(self, state: str, *, error: BaseException | None = None) -> None:
        current = self.payload["state"]
        if state not in _TRANSITIONS[current]:
            raise ValueError(f"invalid run-state transition: {current} -> {state}")
        event = {"state": state, "at": _timestamp()}
        if state == "failed":
            if error is None:
                raise ValueError("failed state requires an error")
            event["error_type"] = type(error).__name__
            event["message"] = str(error)[:1000]
        elif error is not None:
            raise ValueError("errors are valid only for failed transitions")
        self.payload["state"] = state
        self.payload["events"].append(event)
        _write_atomic(self.path, self.payload)


def verify_complete_status(path: Path, protocol_sha256: str) -> dict:
    tracker = RunTracker.open(path.parent)
    if (
        tracker.path != path
        or tracker.payload.get("state") != "complete"
        or tracker.payload.get("protocol_sha256") != protocol_sha256
        or [event.get("state") for event in tracker.payload["events"]]
        != ["planned", "running", "complete"]
    ):
        raise ValueError("run did not complete the required state sequence")
    return tracker.payload
