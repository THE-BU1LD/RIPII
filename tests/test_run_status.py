from __future__ import annotations

import pytest

from ripii.world.run_status import RunTracker, verify_complete_status


def test_run_status_enforces_transition_sequence(tmp_path) -> None:
    tracker = RunTracker.create(tmp_path, run_kind="test", protocol_sha256="a" * 64)
    tracker.transition("running")
    tracker.transition("complete")
    assert verify_complete_status(tmp_path / "status.json", "a" * 64)[
        "state"
    ] == "complete"
    with pytest.raises(ValueError, match="complete -> complete"):
        tracker.transition("complete")


def test_run_status_records_failures_without_tracebacks(tmp_path) -> None:
    tracker = RunTracker.create(tmp_path, run_kind="test", protocol_sha256="b" * 64)
    tracker.transition("running")
    tracker.transition("failed", error=RuntimeError("expected failure"))
    loaded = RunTracker.open(tmp_path)
    assert loaded.payload["state"] == "failed"
    assert loaded.payload["events"][-1]["error_type"] == "RuntimeError"
    with pytest.raises(ValueError, match="required state"):
        verify_complete_status(tmp_path / "status.json", "b" * 64)
