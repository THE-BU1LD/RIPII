from __future__ import annotations

import copy
import json

import pytest
import torch

from ripii.world import closure_study as study
from ripii.world.closure import load_closure_checkpoint
from ripii.world.run_status import _write_atomic


@pytest.fixture(scope="module")
def miniature_phase(tmp_path_factory):
    directory = tmp_path_factory.mktemp("closure_pipeline") / "phase"
    protocol = copy.deepcopy(study.load_protocol())
    protocol["seeds"]["dry_run"] = {"data": 63001, "model": 63011}
    protocol["dataset"].update(train_scenes=2, train_horizon=2, validation_scenes=1,
                               evaluation_scenes=2, evaluation_horizon=2)
    protocol["training"].update(batch_size=2, rollout_horizon=1, validation_every=1,
                                base_updates=2, timing_only_fallback_updates=1)
    previous_threads = torch.get_num_threads()
    previous_rng = torch.random.get_rng_state()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = study.run_phase(directory, protocol, phase="dry_run", updates=1, seconds=120)
    yield directory, payload
    torch.random.set_rng_state(previous_rng)
    torch.use_deterministic_algorithms(previous_deterministic)
    torch.set_num_threads(previous_threads)


def test_complete_pipeline_retains_and_replays_every_arm(miniature_phase):
    directory, payload = miniature_phase
    receipt = study.verify_phase(directory, replay=True)
    assert receipt["raw_split_metrics_verified"] == 44
    assert receipt["checkpoint_split_replays"] == 44
    assert receipt["analysis_regenerated"]
    assert len(payload["units"][0]["arms"]) == 11
    assert json.loads((directory / "status.json").read_text())["state"] == "complete"
    assert json.loads((directory / "h1_results.json").read_text())["corrected_parity_passed"]


def test_global_base_and_compute_checkpoints_are_separate(miniature_phase):
    directory, payload = miniature_phase
    unit = payload["units"][0]
    root = directory / "units" / str(unit["data_seed"])
    _, base = load_closure_checkpoint(root / "global_env_s4/selected.pt")
    _, extended = load_closure_checkpoint(root / "global_compute/selected.pt")
    final = torch.load(root / "global_env_s4/final.pt", weights_only=True)
    assert base["completed_updates"] == 1
    assert extended["completed_updates"] >= 1
    assert final["completed_updates"] == 1
    assert {"optimizer_state", "window_rng_state", "torch_rng_state"} <= set(final)
    assert (root / "global_env_s4/initial.pt").read_bytes() == (root / "global_compute/initial.pt").read_bytes()


def test_hand_computable_metric_weighting_ignores_padding():
    mask = torch.tensor([[True, False], [True, True]])
    target = torch.zeros(2, 3, 2, 6)
    target[..., 4] = 0.1 * mask[:, None]
    target[..., 5] = mask[:, None]
    prediction = target.clone()
    prediction[0, 1:, 0, 0] = 3
    prediction[1, 1:, :, 0] = 1
    # The metric's coordinate errors ignore padding by its explicit mask.
    prediction[0, 1:, 1, 0] = 1000
    data = {"states": target, "mask": mask, "ids": torch.tensor([101, 102])}
    metrics, raw = study.prediction_metrics(prediction, data)
    expected = (22 / 12) ** 0.5
    assert metrics["position_rmse"] == pytest.approx(expected, abs=1e-7)
    assert raw["position_sse_per_scene_time"].tolist() == [[9, 9], [2, 2]]
    assert metrics["velocity_rmse"] == 0


def test_data_split_ids_and_initial_states_are_disjoint(miniature_phase):
    directory, payload = miniature_phase
    manifest = json.loads((directory / "units/63001/data/data_manifest.json").read_text())
    initial = [item for split in manifest["splits"].values() for item in split["initial_scene_sha256"]]
    assert len(initial) == len(set(initial))
    assert all(row["spec"]["license"] == "NOASSERTION" for row in manifest["splits"].values())
    assert payload["phase"] == "dry_run"


def test_no_overwrite_or_implicit_resume(miniature_phase):
    directory, payload = miniature_phase
    with pytest.raises(FileExistsError):
        study.run_phase(directory, payload["protocol"], phase="dry_run", updates=1, seconds=120)


def test_manifest_detects_added_changed_and_unsafe_files(tmp_path):
    (tmp_path / "record.txt").write_text("literal evidence")
    study.write_manifest(tmp_path)
    assert study.verify_manifest(tmp_path)["files_verified"] == 1
    (tmp_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="artifact set"):
        study.verify_manifest(tmp_path)
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "record.txt").write_text("altered evidence")
    with pytest.raises(ValueError, match="integrity"):
        study.verify_manifest(tmp_path)
    (tmp_path / "unsafe").symlink_to(tmp_path / "record.txt")
    with pytest.raises(ValueError, match="symlink"):
        study.write_manifest(tmp_path)


def test_failed_generation_retains_status_and_error(tmp_path, monkeypatch):
    def fail_generation(*_args):
        raise ValueError("deliberate data validation failure")
    monkeypatch.setattr(study, "prepare_data", fail_generation)
    output = tmp_path / "failed"
    protocol = study.load_protocol()
    with pytest.raises(ValueError, match="deliberate"):
        study.run_phase(output, protocol, phase="development", updates=1, seconds=60)
    assert json.loads((output / "status.json").read_text())["state"] == "failed"
    assert not json.loads((output / "phase_partial.json").read_text())["complete"]
    assert study.verify_manifest(output)["files_verified"] > 5
    with pytest.raises(ValueError, match="complete"):
        study.verify_phase(output)


def test_confirmation_requires_freeze_before_data(tmp_path, monkeypatch):
    called = False
    def should_not_generate(*_args):
        nonlocal called
        called = True
        raise AssertionError("protected data accessed")
    monkeypatch.setattr(study, "prepare_data", should_not_generate)
    with pytest.raises(ValueError, match="freeze"):
        study.run_phase(tmp_path / "protected", study.load_protocol(), phase="confirmation",
                        updates=120, seconds=60)
    assert not called


def test_calibration_depends_only_on_times(miniature_phase):
    _, payload = miniature_phase
    before = study.calibration_record(payload)
    altered = copy.deepcopy(payload)
    for arm in altered["units"][0]["arms"].values():
        for metrics in arm["metrics"].values():
            metrics["position_rmse"] = 1000000.0
    after = study.calibration_record(altered)
    before.pop("created_utc")
    after.pop("created_utc")
    assert before == after


def test_budget_and_protocol_reject_invalid_inputs(tmp_path):
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            study.Budget(value)
    clock = study.Budget(1)
    clock.started -= 2
    with pytest.raises(TimeoutError):
        clock.check()
    protocol = study.load_protocol()
    protocol["seeds"]["confirmation"][0]["data"] = protocol["seeds"]["development"][0]["data"]
    path = tmp_path / "protocol.json"
    _write_atomic(path, protocol)
    with pytest.raises(ValueError, match="overlap"):
        study.load_protocol(path)
