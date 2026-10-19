"""Bounded, versioned closure study; see research/closure_study_v1/SPECIFICATION.md.

The CLI separates timing calibration, development, freezing and confirmation.
It never chooses scientific settings from a favorable development outcome.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .closure import ClosureConfig, ClosureModel, load_closure_checkpoint
from .closure_analysis import analyze_phase, render_analysis
from .data import DatasetSpec, _tensor_sha256, validate_dataset
from .experiment import loss_on_states, windows
from .physics import Physics, make_dataset, sample_scene, simulate
from .run_status import RunTracker, _write_atomic, verify_complete_status

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "research/closure_study_v1/protocol.json"
SPLITS = ("test", "more_objects", "composition", "fast")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def source_record() -> dict:
    paths = sorted((ROOT / "ripii").rglob("*.py")) + [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "research/closure_study_v1/SPECIFICATION.md",
        PROTOCOL_PATH,
    ]
    files = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    return {"git_commit": _git("rev-parse", "HEAD"), "files": files,
            "source_sha256": canonical_hash(files)}


def environment_record() -> dict:
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(), "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def stable_environment(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "process_peak_rss_kib"}


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("format") != "ripii-closure-study-protocol-v1":
        raise ValueError("unknown closure study protocol")
    for phase in ("development", "confirmation"):
        seeds = protocol["seeds"][phase]
        if not seeds or len({s["data"] for s in seeds}) != len(seeds):
            raise ValueError("duplicate or missing independent data seeds")
    identities = [protocol["seeds"]["dry_run"]["data"]] + [
        item["data"] for phase in ("development", "confirmation")
        for item in protocol["seeds"][phase]
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("dry/development/confirmation seed overlap")
    if protocol["dataset"]["evaluation_splits"] != list(SPLITS):
        raise ValueError("unexpected evaluation split matrix")
    return protocol


def write_manifest(directory: Path) -> dict:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in evidence: {path}")
        if path.is_file() and path != directory / "manifest.json":
            files[str(path.relative_to(directory))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
    record = {"format": "ripii-closure-evidence-manifest-v1", "files": files}
    _write_atomic(directory / "manifest.json", record)
    return record


def verify_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or unsafe manifest")
    record = json.loads(path.read_text())
    if record.get("format") != "ripii-closure-evidence-manifest-v1":
        raise ValueError("unknown evidence format")
    expected = record.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("empty artifact manifest")
    actual = set()
    for child in directory.rglob("*"):
        if child.is_symlink():
            raise ValueError("symlink in evidence tree")
        if child.is_file() and child != path:
            actual.add(str(child.relative_to(directory)))
    if actual != set(expected):
        raise ValueError("artifact set differs from manifest")
    for relative, info in expected.items():
        member = Path(relative)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("unsafe manifest member")
        child = directory / member
        if sha256(child) != info["sha256"] or child.stat().st_size != info["bytes"]:
            raise ValueError(f"artifact integrity failure: {relative}")
    return {"files_verified": len(expected), "manifest_sha256": sha256(path)}


class Budget:
    def __init__(self, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("budget must be finite and positive")
        self.started = time.perf_counter()
        self.seconds = seconds

    def check(self) -> None:
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError("frozen study wall-time budget exhausted")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def prepare_data(directory: Path, protocol: dict, seed: int) -> tuple[dict, dict]:
    directory.mkdir(parents=True, exist_ok=False)
    cfg = protocol["dataset"]
    physics = Physics(**cfg["physics"])
    datasets, records = {}, {}
    initial_seen: set[str] = set()
    scene_identities: set[int] = set()
    for split in ("train", "validation", *SPLITS):
        scenes = cfg["train_scenes"] if split == "train" else (
            cfg["validation_scenes"] if split == "validation" else cfg["evaluation_scenes"]
        )
        horizon = cfg["train_horizon"] if split == "train" else cfg["evaluation_horizon"]
        spec = DatasetSpec(split, scenes, horizon, seed, cfg["max_objects"], physics)
        data = make_dataset(split, scenes, horizon, seed, cfg["max_objects"], physics)
        validate_dataset(spec, data)
        initial_hashes = []
        for i in range(scenes):
            initial = _tensor_sha256({"state": data["states"][i, 0], "mask": data["mask"][i]})
            if initial in initial_seen:
                raise ValueError("duplicate initial scene within/across splits")
            initial_seen.add(initial)
            initial_hashes.append(initial)
            identity = int(data["ids"][i])
            if identity in scene_identities:
                raise ValueError("duplicate split-qualified scene identity")
            scene_identities.add(identity)
        torch.save(data, directory / f"{split}.pt")
        datasets[split] = data
        records[split] = {"spec": spec.as_record(), "tensor_sha256": _tensor_sha256(data),
                          "initial_scene_sha256": initial_hashes,
                          "identity_policy": "Generator IDs include the fixed split-specific seed-domain offset"}
    _write_atomic(directory / "data_manifest.json", {
        "format": "ripii-closure-data-v1", "data_seed": seed, "splits": records,
        "created_before_training": True,
    })
    return datasets, records


class Oracle(nn.Module):
    def __init__(self, physics: Physics):
        super().__init__()
        self.physics = physics

    def forward(self, state, action, mask):
        return simulate(state, action, mask, self.physics)


def build_model(protocol: dict, arm: dict, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if arm["kind"] == "oracle":
        return Oracle(Physics(**protocol["dataset"]["physics"]))
    physics = protocol["dataset"]["physics"]
    cfg = {**protocol["model"], **{key: physics[key] for key in (
        "dt", "drag", "stiffness", "damping"
    )}, **{key: arm[key] for key in ("kind", "closure", "substeps")}}
    return ClosureModel(ClosureConfig(**cfg))


@torch.no_grad()
def predict(model: nn.Module, data: dict, budget: Budget | None = None) -> torch.Tensor:
    model.eval()
    state = data["states"][:, 0]
    trajectory = [state]
    for step in range(data["actions"].shape[1]):
        if budget:
            budget.check()
        state = model(state, data["actions"][:, step], data["mask"])
        if not torch.isfinite(state).all():
            raise FloatingPointError("nonfinite evaluation trajectory")
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)


def prediction_metrics(prediction: torch.Tensor, data: dict) -> tuple[dict, dict]:
    target, mask = data["states"], data["mask"]
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError("prediction shape/finite contract failed")
    error = prediction[:, 1:, :, :4] - target[:, 1:, :, :4]
    live = mask[:, None, :, None]
    horizon = error.shape[1]
    counts = mask.sum(1)
    pos_sse = (error[..., :2].square() * live).sum((2, 3))
    vel_sse = (error[..., 2:4].square() * live).sum((2, 3))
    denominator = int(counts.sum()) * horizon * 2
    metrics = {
        "position_rmse": float((pos_sse.sum() / denominator).sqrt()),
        "velocity_rmse": float((vel_sse.sum() / denominator).sqrt()),
        "max_abs_position": float(prediction[:, 1:, :, :2].abs().max()),
        "max_abs_velocity": float(prediction[:, 1:, :, 2:4].abs().max()),
        "property_max_error": float((prediction[..., 4:] - target[..., 4:]).abs().max()),
    }
    for h in (1, 4, 16, 32):
        if h <= horizon:
            metrics[f"position_rmse_h{h}"] = float((pos_sse[:, h - 1].sum() / (2 * counts.sum())).sqrt())
    per_scene = (pos_sse.sum(1) / (2 * horizon * counts)).sqrt()
    for n in sorted(set(counts.tolist())):
        selection = counts == n
        metrics[f"position_rmse_n{n}"] = float((pos_sse[selection].sum() / (2 * horizon * counts[selection].sum())).sqrt())
    pred_p = (prediction[:, 1:, :, 2:4] * prediction[:, 1:, :, 5:6] * live).sum(2)
    true_p = (target[:, 1:, :, 2:4] * target[:, 1:, :, 5:6] * live).sum(2)
    metrics["scene_momentum_rmse"] = float((pred_p - true_p).square().mean().sqrt())
    radius = prediction[:, 1:, :, 4:5]
    wall = (prediction[:, 1:, :, :2].abs() + radius - 1).clamp_min(0) * live
    metrics["max_wall_penetration"] = float(wall.max())
    metrics["mean_wall_penetration"] = float(wall.sum() / denominator)
    delta = prediction[:, 1:, :, None, :2] - prediction[:, 1:, None, :, :2]
    distance = delta.square().sum(-1).sqrt()
    radii = prediction[:, 1:, :, 4]
    contact = (radii.unsqueeze(-1) + radii.unsqueeze(-2) - distance).clamp_min(0)
    valid = mask[:, :, None] & mask[:, None, :]
    valid &= torch.triu(torch.ones(mask.shape[1], mask.shape[1], dtype=torch.bool), 1)
    contact = contact * valid[:, None]
    metrics["max_contact_penetration"] = float(contact.max())
    metrics["worst_scene_position_rmse"] = float(per_scene.max())
    metrics["worst_scene_id"] = int(data["ids"][per_scene.argmax()])
    raw = {"position_sse_per_scene_time": pos_sse, "velocity_sse_per_scene_time": vel_sse,
           "per_scene_position_rmse": per_scene, "counts": counts, "ids": data["ids"],
           "momentum_error": pred_p - true_p}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite derived metric")
    return metrics, raw


@torch.no_grad()
def profile_model(model: nn.Module) -> dict:
    result = {}
    for n in (2, 4, 8):
        generator = torch.Generator().manual_seed(36001 + n)
        state, mask = sample_scene(generator, n, 8)
        state, mask = state.unsqueeze(0), mask.unsqueeze(0)
        action = torch.zeros(1, 8, 2)
        for _ in range(2):
            model(state, action, mask)
        started = time.perf_counter()
        for _ in range(10):
            model(state, action, mask)
        result[f"seconds_per_forward_n{n}"] = (time.perf_counter() - started) / 10
    return result


def checkpoint_payload(model: ClosureModel, *, protocol: dict, source: dict,
                       step: int, model_seed: int, validation: float) -> dict:
    return {"format": "ripii-closure-v1", "config": asdict(model.config),
            "model_state": copy.deepcopy(model.state_dict()),
            "completed_updates": step, "seed": model_seed,
            "selection": {"validation_position_rmse": validation},
            "source_sha256": source["source_sha256"], "protocol_sha256": canonical_hash(protocol)}


def train_model(model: ClosureModel, data: dict, directory: Path, protocol: dict,
                source: dict, model_seed: int, updates: int, budget: Budget,
                target_optimization_seconds: float | None = None) -> dict:
    cfg = protocol["training"]
    learning_rate = cfg["law_learning_rate"] if model.config.kind == "law" else cfg["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"])
    generator = torch.Generator().manual_seed(model_seed + cfg["window_seed_offset"])
    initial = copy.deepcopy(model.state_dict())
    torch.save(initial, directory / "initial.pt")
    best_score = math.inf
    best: dict = {}
    history: list[dict] = []
    total_opt = total_validation = 0.0
    base_record: dict | None = None
    base_selected: dict | None = None
    maximum = updates * protocol["compute_control"]["max_update_multiplier"] if target_optimization_seconds else updates
    started = time.perf_counter()
    for step in range(1, maximum + 1):
        budget.check()
        optimization_started = time.perf_counter()
        model.train()
        states, actions, mask = windows(data["train"], cfg["batch_size"], cfg["rollout_horizon"], generator)
        state = states[:, 0]
        optimizer.zero_grad(set_to_none=True)
        loss = state.new_zeros(())
        for t in range(cfg["rollout_horizon"]):
            state = model(state, actions[:, t], mask)
            loss = loss + loss_on_states(state, states[:, t + 1], mask)
        loss = loss / cfg["rollout_horizon"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        total_opt += time.perf_counter() - optimization_started
        at_base = step == updates
        finished = step >= updates and (target_optimization_seconds is None or total_opt >= target_optimization_seconds or step == maximum)
        row = {"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(grad_norm),
               "optimization_seconds": total_opt}
        if step % cfg["validation_every"] == 0 or at_base or finished:
            validation_started = time.perf_counter()
            pred = predict(model, data["validation"], budget)
            score = prediction_metrics(pred, data["validation"])[0]["position_rmse"]
            total_validation += time.perf_counter() - validation_started
            row["validation_position_rmse"] = score
            if score < best_score:
                best_score = score
                best = checkpoint_payload(model, protocol=protocol, source=source,
                                          step=step, model_seed=model_seed, validation=score)
                torch.save(best, directory / "selected.pt")
        history.append(row)
        with (directory / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if at_base:
            base_selected = copy.deepcopy(best)
            base_record = {"completed_updates": step, "optimization_seconds": total_opt,
                           "validation_seconds": total_validation, "selected_update": best["completed_updates"],
                           "validation_position_rmse": best_score,
                           "training_wall_seconds": time.perf_counter() - started}
            if target_optimization_seconds is not None:
                torch.save(base_selected, directory / "base_selected.pt")
                base_final = checkpoint_payload(model, protocol=protocol, source=source,
                                                step=step, model_seed=model_seed,
                                                validation=row["validation_position_rmse"])
                base_final.update({"optimizer_state": copy.deepcopy(optimizer.state_dict()),
                                   "window_rng_state": generator.get_state(),
                                   "torch_rng_state": torch.random.get_rng_state()})
                torch.save(base_final, directory / "base_final.pt")
        if finished:
            break
    final = checkpoint_payload(model, protocol=protocol, source=source,
                               step=step, model_seed=model_seed,
                               validation=row["validation_position_rmse"])
    final.update({"optimizer_state": optimizer.state_dict(), "window_rng_state": generator.get_state(),
                  "torch_rng_state": torch.random.get_rng_state(),
                  "best_validation_position_rmse": best_score, "selected_update": best["completed_updates"]})
    torch.save(final, directory / "final.pt")
    model.load_state_dict(best["model_state"])
    return {"completed_updates": step, "optimization_seconds": total_opt,
            "validation_seconds": total_validation, "training_wall_seconds": time.perf_counter() - started,
            "selected_update": best["completed_updates"], "validation_position_rmse": best_score,
            "base_record": base_record, "base_selected": base_selected,
            "compute_coverage_ratio": total_opt / target_optimization_seconds if target_optimization_seconds else None,
            "training_points": len(history)}


def evaluate_arm(model: nn.Module, data: dict, directory: Path, protocol: dict,
                 data_seed: int, budget: Budget) -> dict:
    started = time.perf_counter()
    metrics, raw, predictions = {}, {}, {}
    for split in SPLITS:
        pred = predict(model, data[split], budget)
        metrics[split], raw[split] = prediction_metrics(pred, data[split])
        predictions[split] = pred
        torch.save(predictions, directory / "predictions.pt")
        torch.save(raw, directory / "raw_metrics.pt")
        _write_atomic(directory / "evaluation_progress.json", {
            "complete": False, "completed_splits": list(metrics), "metrics": metrics,
        })
    noisy = {key: value.clone() for key, value in data["test"].items()}
    generator = torch.Generator().manual_seed(data_seed + 8000000)
    noise = torch.randn(noisy["states"][:, 0, :, :4].shape, generator=generator)
    noisy["states"][:, 0, :, :4] += protocol["dataset"]["initial_noise_std"] * noise * noisy["mask"].unsqueeze(-1)
    noisy_prediction = predict(model, noisy, budget)
    noise_metrics, noise_raw = prediction_metrics(noisy_prediction, data["test"])
    predictions["noisy_test"] = noisy_prediction
    raw["noisy_test"] = noise_raw
    torch.save(predictions, directory / "predictions.pt")
    torch.save(raw, directory / "raw_metrics.pt")
    result = {"metrics": metrics, "noise_metrics": noise_metrics,
              "profile": profile_model(model), "evaluation_seconds": time.perf_counter() - started,
              "prediction_sha256": _tensor_sha256(predictions)}
    _write_atomic(directory / "evaluation.json", result)
    _write_atomic(directory / "evaluation_progress.json", {
        "complete": True, "completed_splits": [*SPLITS, "noisy_test"],
    })
    return result


def _new_arm(directory: Path, arm: dict, protocol_hash: str) -> RunTracker:
    directory.mkdir(parents=True, exist_ok=False)
    _write_atomic(directory / "arm.json", arm)
    tracker = RunTracker.create(directory, run_kind="closure-study-arm", protocol_sha256=protocol_hash)
    tracker.transition("running")
    return tracker


def run_unit(directory: Path, protocol: dict, source: dict, seed: dict,
             data: dict, updates: int, budget: Budget) -> dict:
    result = {"data_seed": seed["data"], "model_seed": seed["model"], "arms": {}}
    protocol_hash = canonical_hash(protocol)
    for arm in protocol["arms"]:
        budget.check()
        name = arm["id"]
        arm_directory = directory / name
        tracker = _new_arm(arm_directory, arm, protocol_hash)
        compute_tracker: RunTracker | None = None
        model = None
        try:
            model = build_model(protocol, arm, seed["model"])
            parameters = sum(parameter.numel() for parameter in model.parameters())
            target = None
            train_directory = arm_directory
            if name == protocol["compute_control"]["source_arm"]:
                target = result["arms"][protocol["compute_control"]["target_arm"]]["optimization_seconds"]
                train_directory = directory / protocol["compute_control"]["id"]
                compute_tracker = _new_arm(train_directory, {**arm, "id": protocol["compute_control"]["id"],
                                                          "target_optimization_seconds": target}, protocol_hash)
            if parameters:
                assert isinstance(model, ClosureModel)
                training = train_model(model, data, train_directory, protocol, source,
                                       seed["model"], updates, budget, target)
            else:
                training = {"completed_updates": 0, "optimization_seconds": 0.0,
                            "validation_seconds": 0.0, "training_wall_seconds": 0.0,
                            "selected_update": 0, "validation_position_rmse": None,
                            "base_record": None, "base_selected": None,
                            "compute_coverage_ratio": None, "training_points": 0}
                if isinstance(model, ClosureModel):
                    torch.save(checkpoint_payload(model, protocol=protocol, source=source,
                                                  step=0, model_seed=seed["model"], validation=0.0),
                               arm_directory / "selected.pt")
                else:
                    _write_atomic(arm_directory / "oracle.json", {"physics": protocol["dataset"]["physics"],
                                                                  "status": "known-law reference, not a learned contender"})
            base_selected = training.pop("base_selected")
            base_record = training.pop("base_record")
            if target is not None:
                extended = {"parameters": parameters, **training,
                            **evaluate_arm(model, data, train_directory, protocol, seed["data"], budget)}
                result["arms"][protocol["compute_control"]["id"]] = extended
                _write_atomic(train_directory / "result.json", extended)
                assert compute_tracker is not None and base_selected is not None and base_record is not None
                compute_tracker.transition("complete")
                model.load_state_dict(base_selected["model_state"])
                torch.save(base_selected, arm_directory / "selected.pt")
                shutil.copyfile(train_directory / "base_final.pt", arm_directory / "final.pt")
                shutil.copyfile(train_directory / "initial.pt", arm_directory / "initial.pt")
                training = {**training, **base_record, "compute_coverage_ratio": None,
                            "training_history": f"../{protocol['compute_control']['id']}/training.jsonl",
                            "training_points": updates}
            record = {"parameters": parameters, **training,
                      **evaluate_arm(model, data, arm_directory, protocol, seed["data"], budget)}
            if isinstance(model, ClosureModel) and hasattr(model, "learned_coupling"):
                coefficient = model.learned_coupling()
                if coefficient is not None:
                    record["fitted_coupling"] = float(coefficient.detach())
            result["arms"][name] = record
            _write_atomic(arm_directory / "result.json", record)
            tracker.transition("complete")
            _write_atomic(directory / "unit.json", result)
            print(json.dumps({"event": "arm_complete", "data_seed": seed["data"], "arm": name,
                              "updates": record["completed_updates"], "elapsed_seconds": budget.elapsed}), flush=True)
        except BaseException as exc:
            if isinstance(model, ClosureModel):
                torch.save({"format": "ripii-closure-failed-state-v1", "config": asdict(model.config),
                            "model_state": model.state_dict(), "error_type": type(exc).__name__,
                            "usable_as_selected_checkpoint": False}, arm_directory / "failed_state.pt")
            tracker.transition("failed", error=exc)
            if compute_tracker is not None and compute_tracker.payload["state"] in ("planned", "running"):
                compute_tracker.transition("failed", error=exc)
            _write_atomic(directory / "unit_partial.json", result)
            raise
    return result


def _snapshot(directory: Path, source: dict) -> None:
    for relative in source["files"]:
        destination = directory / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    _write_atomic(directory / "source.json", source)


@torch.no_grad()
def limiting_case_audit(directory: Path) -> dict:
    fixtures: list[dict] = [
        {"id": "free", "state": [0.0, 0.0, 0.3, -0.2, 0.07, 0.7],
         "action": [0.0, 0.0], "drag": 0.0, "stiffness": 0.0, "damping": 0.0},
        {"id": "constant_force", "state": [0.0, 0.0, 0.2, -0.1, 0.07, 0.7],
         "action": [0.8, -0.3], "drag": 0.0, "stiffness": 0.0, "damping": 0.0},
        {"id": "drag_light", "state": [0.0, 0.0, 0.3, -0.2, 0.07, 0.7],
         "action": [0.0, 0.0], "drag": 0.06, "stiffness": 100.0, "damping": 1.5},
        {"id": "drag_heavy", "state": [0.0, 0.0, 0.3, -0.2, 0.07, 1.4],
         "action": [0.0, 0.0], "drag": 0.06, "stiffness": 100.0, "damping": 1.5},
        {"id": "wall", "state": [0.95, 0.3, 0.65, -0.1, 0.07, 0.7],
         "action": [0.0, 0.0], "drag": 0.06, "stiffness": 100.0, "damping": 1.5},
    ]
    audit_protocol = {"format": "ripii-closure-limiting-cases-v1", "dtype": "float64",
                      "steps": 8, "dt": 0.05, "target_substeps": 4, "fixtures": fixtures,
                      "parity_max_absolute_tolerance": 2e-12, "training_updates": 0}
    _write_atomic(directory / "h1_protocol.json", audit_protocol)
    rows = []
    for fixture in fixtures:
        state = torch.tensor([[fixture["state"]]], dtype=torch.float64)
        action = torch.tensor([[fixture["action"]]], dtype=torch.float64)
        mask = torch.ones(1, 1, dtype=torch.bool)
        constants = {key: fixture[key] for key in ("drag", "stiffness", "damping")}
        physics = Physics(**constants)
        target = [state]
        for _ in range(8):
            target.append(simulate(target[-1], action, mask, physics))
        truth = torch.stack(target, 1)
        for closure in (False, True):
            for substeps in (1, 4):
                with torch.random.fork_rng(devices=[]):
                    model = ClosureModel(ClosureConfig(closure=closure, substeps=substeps, **constants)).double()
                prediction = model.rollout(state, action[:, None].expand(-1, 8, -1, -1), mask)
                error = prediction[:, 1:, :, :4] - truth[:, 1:, :, :4]
                row: dict = {"fixture": fixture["id"], "closure": closure, "substeps": substeps,
                       "maximum_absolute_error": float(error.abs().max()),
                       "position_rmse": float(error[..., :2].square().mean().sqrt()),
                       "velocity_rmse": float(error[..., 2:4].square().mean().sqrt()),
                       "prediction": prediction.tolist(), "target": truth.tolist()}
                rows.append(row)
                if closure and substeps == 4 and row["maximum_absolute_error"] > 2e-12:
                    raise AssertionError("corrected single-disc parity failed")
    result = {"format": "ripii-closure-h1-results-v1", "rows": rows,
              "corrected_parity_passed": True, "scope": "literal single-disc cases, float64, eight observations"}
    _write_atomic(directory / "h1_results.json", result)
    return result


def run_phase(output: Path, protocol: dict, *, phase: str, updates: int,
              seconds: float, freeze: dict | None = None) -> dict:
    if phase not in ("dry_run", "development", "confirmation"):
        raise ValueError("unknown research phase")
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(seconds)
    source = source_record()
    protocol_hash = canonical_hash(protocol)
    if phase == "confirmation":
        if not freeze or freeze.get("format") != "ripii-closure-confirmation-freeze-v1":
            raise ValueError("confirmation requires a valid pre-outcome freeze")
        if (freeze["source"]["source_sha256"] != source["source_sha256"]
                or freeze["source"]["git_commit"] != source["git_commit"]
                or freeze["protocol_sha256"] != protocol_hash or freeze["base_updates"] != updates):
            raise ValueError("confirmation source/protocol/update budget differs from freeze")
        if _git("status", "--porcelain"):
            raise ValueError("confirmation requires the frozen clean Git tree")
        if stable_environment(freeze["environment"]) != stable_environment(environment_record()):
            raise ValueError("confirmation runtime environment differs from freeze")
        if freeze["confirmation_seeds"] != protocol["seeds"]["confirmation"]:
            raise ValueError("confirmation seed receipt differs from protocol")
        _write_atomic(output / "confirmation_freeze.json", freeze)
    _write_atomic(output / "protocol.json", protocol)
    _write_atomic(output / "execution.json", {"phase": phase, "base_updates": updates,
                                              "wall_budget_seconds": seconds, "created_utc": timestamp()})
    _snapshot(output, source)
    _write_atomic(output / "environment.json", environment_record())
    freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "environment_freeze.txt").write_text(freeze_output)
    tracker = RunTracker.create(output, run_kind=f"closure-study-{phase}", protocol_sha256=protocol_hash)
    tracker.transition("running")
    units: list[dict] = []
    payload = {"format": "ripii-closure-phase-results-v1", "phase": phase,
               "protocol": protocol, "units": units, "complete": False}
    try:
        if phase == "dry_run":
            limiting_case_audit(output)
        seeds = [protocol["seeds"]["dry_run"]] if phase == "dry_run" else protocol["seeds"][phase]
        prepared = []
        data_records = {}
        # Freeze every dataset identity before the first optimization/evaluation.
        for seed in seeds:
            budget.check()
            unit_directory = output / "units" / str(seed["data"])
            unit_directory.mkdir(parents=True)
            data, data_record = prepare_data(unit_directory / "data", protocol, seed["data"])
            data_records[str(seed["data"])] = data_record
            prepared.append((seed, unit_directory, data))
        _write_atomic(output / "dataset_freeze.json", {"format": "ripii-closure-dataset-freeze-v1",
                                                       "created_utc": timestamp(), "before_first_training": True,
                                                       "source_sha256": source["source_sha256"],
                                                       "protocol_sha256": protocol_hash, "units": data_records})
        for seed, unit_directory, data in prepared:
            units.append(run_unit(unit_directory, protocol, source, seed, data, updates, budget))
            _write_atomic(output / "phase_partial.json", payload)
        if source_record()["source_sha256"] != source["source_sha256"]:
            raise ValueError("scientific sources changed during execution")
        payload["complete"] = True
        payload["wall_seconds"] = budget.elapsed
        payload["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write_atomic(output / "phase_results.json", payload)
        render_analysis(payload, output / "analysis")
        payload["wall_seconds"] = budget.elapsed
        payload["wall_time_scope"] = "phase setup, data, training, evaluation and analysis; excludes final manifest hashing"
        _write_atomic(output / "phase_results.json", payload)
        budget.check()
        tracker.transition("complete")
    except BaseException as exc:
        payload["wall_seconds"] = budget.elapsed
        payload["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_atomic(output / "phase_partial.json", payload)
        tracker.transition("failed", error=exc)
        write_manifest(output)
        raise
    write_manifest(output)
    return payload


def calibration_record(payload: dict) -> dict:
    if payload["phase"] != "dry_run" or not payload["complete"] or len(payload["units"]) != 1:
        raise ValueError("calibration requires the completed dry-run grid")
    protocol = payload["protocol"]
    arms = payload["units"][0]["arms"]
    projections = {}
    for updates in (protocol["training"]["base_updates"], protocol["training"]["timing_only_fallback_updates"]):
        cost = 0.0
        for arm in protocol["arms"]:
            record = arms[arm["id"]]
            steps = record["completed_updates"]
            if steps:
                cost += updates * record["optimization_seconds"] / steps
                validations = math.ceil(updates / protocol["training"]["validation_every"])
                cost += validations * record["validation_seconds"]
            cost += record["evaluation_seconds"]
        hierarchy = arms[protocol["compute_control"]["target_arm"]]
        flat = arms[protocol["compute_control"]["source_arm"]]
        extra = updates * max(0.0, hierarchy["optimization_seconds"] / hierarchy["completed_updates"]
                              - flat["optimization_seconds"] / flat["completed_updates"])
        cost += extra + arms[protocol["compute_control"]["id"]]["evaluation_seconds"]
        # Include measured data/metadata overhead, plus extra control validation.
        measured_accounted = sum((r.get("optimization_seconds", 0) + r.get("validation_seconds", 0)
                                  + r.get("evaluation_seconds", 0)) for key, r in arms.items()
                                 if key != protocol["compute_control"]["source_arm"])
        overhead = max(0.0, payload["wall_seconds"] - measured_accounted)
        cost += overhead + 3 * math.ceil(updates / 40) * flat["validation_seconds"]
        repetitions = len(protocol["seeds"]["development"]) + len(protocol["seeds"]["confirmation"])
        projections[str(updates)] = cost * repetitions * protocol["budget"]["projection_safety_factor"]
    choice = next((int(key) for key, value in projections.items()
                   if value <= protocol["budget"]["projection_threshold_seconds"]), None)
    return {"format": "ripii-closure-timing-calibration-v1", "created_utc": timestamp(),
            "selected_base_updates": choice, "projected_grid_seconds": projections,
            "selection_basis": "timing only; no accuracy-based selection",
            "protocol_sha256": canonical_hash(protocol), "dry_run_complete": True}


def _read_completed(directory: Path) -> tuple[dict, dict]:
    receipt = verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    verify_complete_status(directory / "status.json", canonical_hash(protocol))
    payload = json.loads((directory / "phase_results.json").read_text())
    if not payload["complete"] or canonical_hash(payload["protocol"]) != canonical_hash(protocol):
        raise ValueError("study phase is incomplete")
    return payload, receipt


def read_calibration(path: Path, protocol: dict) -> dict:
    if path.name != "calibration.json":
        raise ValueError("calibration must be the retained dry-run artifact")
    payload, receipt = _read_completed(path.parent)
    if payload["phase"] != "dry_run":
        raise ValueError("calibration must come from a completed dry run")
    record = json.loads(path.read_text())
    recomputed = calibration_record(payload)
    # Timestamp records creation, while every decision-bearing field is derived.
    for key in recomputed:
        if key != "created_utc" and record.get(key) != recomputed[key]:
            raise ValueError("calibration is not the timing-only derived decision")
    if record["protocol_sha256"] != canonical_hash(protocol):
        raise ValueError("calibration protocol mismatch")
    choice = record["selected_base_updates"]
    if choice not in {protocol["training"]["base_updates"], protocol["training"]["timing_only_fallback_updates"]}:
        raise TimeoutError("timing projection exceeds the frozen budget")
    return {**record, "dry_run_manifest_sha256": receipt["manifest_sha256"]}


def create_freeze(development: Path, calibration: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite a confirmation freeze")
    payload, receipt = _read_completed(development)
    if payload["phase"] != "development":
        raise ValueError("confirmation must follow the development phase")
    protocol = load_protocol()
    calibrate = read_calibration(calibration, protocol)
    if (canonical_hash(payload["protocol"]) != canonical_hash(protocol)
            or calibrate["protocol_sha256"] != canonical_hash(protocol)):
        raise ValueError("development/calibration protocol mismatch")
    execution = json.loads((development / "execution.json").read_text())
    if execution["base_updates"] != calibrate["selected_base_updates"]:
        raise ValueError("development executed a different update budget from calibration")
    development_verification = verify_phase(development)
    if _git("status", "--porcelain"):
        raise ValueError("commit the complete reviewed implementation before confirmation freeze")
    source = source_record()
    previous_source = json.loads((development / "source.json").read_text())
    if previous_source["source_sha256"] != source["source_sha256"]:
        raise ValueError("scientific sources changed after development; v1 permits no scientific revision")
    spent = payload["wall_seconds"]
    remaining = protocol["budget"]["outcome_bearing_seconds"] - spent
    if remaining <= 0:
        raise TimeoutError("no confirmation budget remains")
    result = {"format": "ripii-closure-confirmation-freeze-v1", "created_utc": timestamp(),
              "protocol_sha256": canonical_hash(protocol), "source": source,
              "environment": environment_record(), "base_updates": calibrate["selected_base_updates"],
              "development_manifest_sha256": receipt["manifest_sha256"],
              "development_verification": development_verification,
              "dry_run_manifest_sha256": calibrate["dry_run_manifest_sha256"],
              "development_wall_seconds": spent, "remaining_wall_seconds": remaining,
              "confirmation_seeds": protocol["seeds"]["confirmation"],
              "eligibility": "complete valid pipeline, independent of favorable development accuracy",
              "confirmation_data_state": "not generated or inspected"}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output, result)
    return result


def verify_phase(directory: Path, *, replay: bool = False) -> dict:
    payload, receipt = _read_completed(directory)
    protocol = payload["protocol"]
    expected_seeds = ([protocol["seeds"]["dry_run"]] if payload["phase"] == "dry_run"
                      else protocol["seeds"][payload["phase"]])
    if [(u["data_seed"], u["model_seed"]) for u in payload["units"]] != [
            (s["data"], s["model"]) for s in expected_seeds]:
        raise ValueError("independent-unit grid mismatch")
    expected_arms = {arm["id"] for arm in protocol["arms"]} | {protocol["compute_control"]["id"]}
    raw_checked = replayed = 0
    model: nn.Module
    source = json.loads((directory / "source.json").read_text())
    if replay and source_record()["source_sha256"] != source["source_sha256"]:
        raise ValueError("checkpoint replay requires the retained scientific source version")
    execution = json.loads((directory / "execution.json").read_text())
    for unit in payload["units"]:
        if set(unit["arms"]) != expected_arms:
            raise ValueError("arm grid mismatch")
        unit_dir = directory / "units" / str(unit["data_seed"])
        data_record = json.loads((unit_dir / "data/data_manifest.json").read_text())
        data = {}
        for split in ("train", "validation", *SPLITS):
            tensors = torch.load(unit_dir / "data" / f"{split}.pt", weights_only=True)
            if _tensor_sha256(tensors) != data_record["splits"][split]["tensor_sha256"]:
                raise ValueError("dataset tensor hash mismatch")
            data[split] = tensors
        for name, record in unit["arms"].items():
            arm_dir = unit_dir / name
            verify_complete_status(arm_dir / "status.json", canonical_hash(protocol))
            if name != "oracle":
                loaded, checkpoint = load_closure_checkpoint(arm_dir / "selected.pt")
                original_name = (protocol["compute_control"]["source_arm"]
                                 if name == protocol["compute_control"]["id"] else name)
                arm = next(a for a in protocol["arms"] if a["id"] == original_name)
                with torch.random.fork_rng(devices=[]):
                    expected_model = build_model(protocol, arm, unit["model_seed"])
                if (not isinstance(expected_model, ClosureModel)
                        or asdict(loaded.config) != asdict(expected_model.config)
                        or checkpoint["seed"] != unit["model_seed"]
                        or checkpoint["source_sha256"] != source["source_sha256"]
                        or checkpoint["protocol_sha256"] != canonical_hash(protocol)
                        or checkpoint["completed_updates"] != record["selected_update"]
                        or sum(p.numel() for p in loaded.parameters()) != record["parameters"]):
                    raise ValueError("checkpoint/config/source/unit identity mismatch")
                if record["parameters"]:
                    expected_steps = execution["base_updates"]
                    if name == protocol["compute_control"]["id"]:
                        if not expected_steps <= record["completed_updates"] <= expected_steps * protocol["compute_control"]["max_update_multiplier"]:
                            raise ValueError("compute-control update budget mismatch")
                    elif record["completed_updates"] != expected_steps:
                        raise ValueError("base-arm update budget mismatch")
            predictions = torch.load(arm_dir / "predictions.pt", weights_only=True)
            if _tensor_sha256(predictions) != record["prediction_sha256"]:
                raise ValueError("prediction tensor hash mismatch")
            for split in SPLITS:
                recomputed, raw = prediction_metrics(predictions[split], data[split])
                if recomputed != record["metrics"][split]:
                    raise ValueError(f"raw prediction metric mismatch: {name}/{split}")
                retained = torch.load(arm_dir / "raw_metrics.pt", weights_only=True)[split]
                if set(raw) != set(retained) or not all(torch.equal(raw[k], retained[k]) for k in raw):
                    raise ValueError("per-scene raw statistic mismatch")
                raw_checked += 1
            noise_metrics, _ = prediction_metrics(predictions["noisy_test"], data["test"])
            if noise_metrics != record["noise_metrics"]:
                raise ValueError("noise metric mismatch")
            if replay:
                if name == "oracle":
                    model = Oracle(Physics(**protocol["dataset"]["physics"]))
                else:
                    model = loaded
                for split in SPLITS:
                    regenerated = predict(model, data[split])
                    if not torch.equal(regenerated, predictions[split]):
                        raise ValueError(f"checkpoint trajectory replay mismatch: {name}/{split}")
                    replayed += 1
    analysis = json.loads((directory / "analysis/analysis.json").read_text())
    analysis.pop("artifacts", None)
    if analysis != analyze_phase(payload):
        raise ValueError("statistical analysis/decision does not regenerate from phase results")
    if (not analysis["complete"] or analysis["validation"]["errors"]
            or analysis["validation"]["missing"]):
        raise ValueError("analysis reports an invalid or incomplete measurement grid")
    return {**receipt, "raw_split_metrics_verified": raw_checked,
            "checkpoint_split_replays": replayed,
            "analysis_regenerated": True,
            "scope": "integrity, grid, config/source/seed identities, raw metrics and statistical analysis; optional deterministic checkpoint replay; not an external hardware reproduction"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run")
    dry.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--phase", choices=("development", "confirmation"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--calibration", type=Path)
    run.add_argument("--freeze", type=Path)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--development", type=Path, required=True)
    freeze_parser.add_argument("--calibration", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("directory", type=Path)
    verify_parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_rng = torch.random.get_rng_state()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        protocol = load_protocol()
        if args.command == "dry-run":
            payload = run_phase(args.output, protocol, phase="dry_run", updates=6,
                                seconds=protocol["budget"]["dry_run_seconds"])
            calibrated = calibration_record(payload)
            _write_atomic(args.output / "calibration.json", calibrated)
            write_manifest(args.output)
            print(json.dumps(calibrated), flush=True)
        elif args.command == "freeze":
            print(json.dumps(create_freeze(args.development, args.calibration, args.output)), flush=True)
        elif args.command == "verify":
            print(json.dumps(verify_phase(args.directory, replay=args.replay)), flush=True)
        elif args.phase == "development":
            if args.calibration is None:
                raise ValueError("development requires timing calibration")
            calibration = read_calibration(args.calibration, protocol)
            updates = calibration["selected_base_updates"]
            if updates is None:
                raise TimeoutError("timing projection exceeds the frozen budget")
            # Reserve confirmation's proportional share; a slow development grid
            # must not consume the entire authorized scientific allocation.
            development_share = len(protocol["seeds"]["development"]) / (
                len(protocol["seeds"]["development"]) + len(protocol["seeds"]["confirmation"]))
            run_phase(args.output, protocol, phase="development", updates=updates,
                      seconds=protocol["budget"]["outcome_bearing_seconds"] * development_share)
        else:
            if args.freeze is None:
                raise ValueError("confirmation requires a recorded freeze")
            frozen = json.loads(args.freeze.read_text())
            run_phase(args.output, protocol, phase="confirmation", updates=frozen["base_updates"],
                      seconds=frozen["remaining_wall_seconds"], freeze=frozen)
        return 0
    finally:
        torch.random.set_rng_state(previous_rng)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(main())
