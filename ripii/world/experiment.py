from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..utils.statistics import descriptive_summary, paired_sign_flip_test
from .data import DatasetSpec, load_dataset
from .models import VARIANTS, WorldModel
from .physics import Physics
from .protocol import ExperimentProtocol
from .run_status import RunTracker, verify_complete_status


@dataclass
class Experiment:
    steps: int = 300
    train_scenes: int = 192
    eval_scenes: int = 32
    train_horizon: int = 16
    test_horizon: int = 32
    rollout_steps: int = 4
    batch_size: int = 32
    hidden: int = 64
    max_objects: int = 8
    lr: float = 0.001
    data_seed: int = 2026
    validate_every: int = 50
    quantizer_weight: float = 0.01
    global_coupling: float = 0.0

    def validate(self):
        for key, value in asdict(self).items():
            if key not in {
                "lr",
                "quantizer_weight",
                "global_coupling",
                "data_seed",
            } and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.data_seed, int):
            raise ValueError("data_seed must be an integer")
        if (
            not math.isfinite(self.lr)
            or not 0 < self.lr < 1
            or not math.isfinite(self.quantizer_weight)
            or self.quantizer_weight < 0
            or not math.isfinite(self.global_coupling)
            or self.global_coupling < 0
        ):
            raise ValueError(
                "invalid learning rate, quantizer weight, or global coupling"
            )
        if (
            self.rollout_steps > self.train_horizon
            or self.test_horizon < self.rollout_steps
        ):
            raise ValueError("rollout length exceeds trajectory horizon")
        if not 5 <= self.max_objects <= 16 or self.hidden % 4:
            raise ValueError("max_objects must be 5..16 and hidden a multiple of 4")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _finite_json(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))


def _parse_json_object(content: str, label: str) -> dict:
    try:
        payload = json.loads(content, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict) or not _finite_json(payload):
        raise ValueError(f"{label} must be a finite JSON object")
    return payload


def _load_json_object(path: Path, label: str) -> dict:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    return _parse_json_object(content, label)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path: str | Path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "ripii-world-v1"
        or not isinstance(checkpoint.get("model_spec"), dict)
        or not isinstance(checkpoint.get("model"), dict)
        or not isinstance(checkpoint.get("completed_steps"), int)
        or isinstance(checkpoint.get("completed_steps"), bool)
        or checkpoint["completed_steps"] < 1
        or not isinstance(checkpoint.get("best_validation"), (int, float))
        or isinstance(checkpoint.get("best_validation"), bool)
        or not math.isfinite(checkpoint["best_validation"])
        or checkpoint["best_validation"] < 0
        or not checkpoint["model"]
        or not all(
            isinstance(name, str)
            and isinstance(value, torch.Tensor)
            and torch.isfinite(value).all()
            for name, value in checkpoint["model"].items()
        )
    ):
        raise ValueError("not a RIPII world-model checkpoint")
    model = WorldModel(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def loss_on_states(prediction, target, mask):
    if prediction.shape != target.shape or prediction.shape[:2] != mask.shape:
        raise ValueError("prediction, target, and mask shapes do not align")
    if not mask.any(dim=1).all():
        raise ValueError("loss requires at least one live object per scene")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("non-finite loss input")
    weight = prediction.new_tensor([4.0, 4.0, 1.0, 1.0])
    error = (prediction[..., :4] - target[..., :4]).square() * weight
    return (error * mask.unsqueeze(-1)).sum() / (mask.sum().clamp_min(1) * weight.sum())


def windows(data, batch_size, horizon, generator):
    if batch_size < 1 or horizon < 1 or horizon > data["actions"].shape[1]:
        raise ValueError("invalid batch size or rollout horizon")
    scenes = torch.randint(data["states"].shape[0], (batch_size,), generator=generator)
    starts = torch.randint(
        data["actions"].shape[1] - horizon + 1, (batch_size,), generator=generator
    )
    offsets = torch.arange(horizon + 1)
    states = data["states"][scenes[:, None], starts[:, None] + offsets]
    actions = data["actions"][scenes[:, None], starts[:, None] + offsets[:-1]]
    mask = data["mask"][scenes]
    # Permute live and padded slots alike; no model gets a stable object identity index.
    permutation = torch.rand(batch_size, mask.shape[1], generator=generator).argsort(-1)
    states = states.gather(
        2, permutation[:, None, :, None].expand(-1, horizon + 1, -1, 6)
    )
    actions = actions.gather(
        2, permutation[:, None, :, None].expand(-1, horizon, -1, 2)
    )
    return states, actions, mask.gather(1, permutation)


@torch.no_grad()
def evaluate(model, data, horizons=(1, 4, 16, 32)):
    model.eval()
    start = time.perf_counter()
    current = data["states"][:, 0]
    predicted_states = [current]
    diagnostic_rows: list[dict[str, float]] = []
    assignment_history = []
    for step in range(data["actions"].shape[1]):
        current = model(current, data["actions"][:, step], data["mask"])
        predicted_states.append(current)
        if hasattr(model, "diagnostics"):
            diagnostic_rows.append(model.diagnostics(data["mask"]))
        assignment = getattr(model, "last_assignments", None)
        if assignment is not None:
            assignment_history.append(assignment.detach())
    predicted = torch.stack(predicted_states, dim=1)
    elapsed = time.perf_counter() - start
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("non-finite world-model rollout")
    mask = data["mask"]
    error = predicted[:, 1:, :, :4] - data["states"][:, 1:, :, :4]
    live = mask[:, None, :, None]
    denom = mask.sum() * error.shape[1] * 2
    result = {
        "position_rmse": float(((error[..., :2].square() * live).sum() / denom).sqrt()),
        "velocity_rmse": float(
            ((error[..., 2:4].square() * live).sum() / denom).sqrt()
        ),
        "rollout_seconds": elapsed,
        "milliseconds_per_scene_step": 1000 * elapsed / (len(mask) * error.shape[1]),
    }
    for horizon in sorted(set(horizons) | {error.shape[1]}):
        if horizon <= error.shape[1]:
            result[f"position_rmse_h{horizon}"] = float(
                (
                    (error[:, horizon - 1, :, :2].square() * mask[..., None]).sum()
                    / (mask.sum() * 2)
                ).sqrt()
            )
    per_scene = (
        (error[..., :2].square() * live).sum((1, 2, 3))
        / (mask.sum(1) * error.shape[1] * 2)
    ).sqrt()
    result["worst_scene_position_rmse"] = float(per_scene.max())
    result["worst_scene_id"] = int(data["ids"][per_scene.argmax()])
    for quantile in (0.5, 0.9, 0.95):
        result[f"per_scene_position_rmse_p{int(quantile * 100)}"] = float(
            torch.quantile(per_scene, quantile)
        )
    first_error = error[:, 0, :, :2]
    last_error = error[:, -1, :, :2]
    first_rmse = ((first_error.square() * mask[..., None]).sum() / (mask.sum() * 2)).sqrt()
    last_rmse = ((last_error.square() * mask[..., None]).sum() / (mask.sum() * 2)).sqrt()
    result["position_rmse_growth_final_over_h1"] = float(
        last_rmse / first_rmse.clamp_min(1e-12)
    )
    result["max_abs_predicted_position"] = float(
        predicted[:, 1:, :, :2]
        .masked_select(mask[:, None, :, None])
        .abs()
        .max()
    )
    result["max_abs_predicted_velocity"] = float(
        predicted[:, 1:, :, 2:4]
        .masked_select(mask[:, None, :, None])
        .abs()
        .max()
    )
    property_error = predicted[:, 1:, :, 4:] - data["states"][:, 1:, :, 4:]
    result["max_abs_property_drift"] = float(
        property_error.masked_select(mask[:, None, :, None]).abs().max()
    )
    predicted_momentum = (
        predicted[:, 1:, :, 2:4] * predicted[:, 1:, :, 5:6] * live
    ).sum(2)
    target_momentum = (
        data["states"][:, 1:, :, 2:4]
        * data["states"][:, 1:, :, 5:6]
        * live
    ).sum(2)
    result["scene_momentum_rmse"] = float(
        (predicted_momentum - target_momentum).square().mean().sqrt()
    )
    outside = (
        predicted[:, 1:, :, :2].abs() > 1.0 - predicted[:, 1:, :, 4:5]
    ).any(-1)
    result["outside_arena_fraction"] = float(
        (outside * mask[:, None]).sum() / (mask.sum() * error.shape[1])
    )
    result["outside_arena_scene_fraction"] = float(
        (outside & mask[:, None]).any(dim=(1, 2)).float().mean()
    )
    diagnostic_keys = sorted({key for row in diagnostic_rows for key in row})
    for key in diagnostic_keys:
        values = [row[key] for row in diagnostic_rows if key in row]
        result[key] = sum(values) / len(values)
    if len(assignment_history) >= 2:
        changes = [
            (current[mask] - previous[mask]).abs().mean()
            for previous, current in zip(assignment_history, assignment_history[1:])
        ]
        result["assignment_temporal_change"] = float(torch.stack(changes).mean())
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite world-model evaluation metric")
    return result


def train(
    cfg: Experiment,
    output: Path,
    variant="multiscale",
    seed=3,
    bottleneck="continuous",
    hidden=None,
    resume: Path | None = None,
):
    cfg.validate()
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(f"refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = WorldModel(
        variant, hidden or cfg.hidden, cfg.max_objects, bottleneck=bottleneck
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(seed + 700_001)
    first_step, best, elapsed_before = 0, float("inf"), 0.0
    if resume:
        model, saved = load_model(resume)
        if (
            not isinstance(saved.get("experiment"), dict)
            or not isinstance(saved.get("optimizer"), dict)
            or not isinstance(saved.get("sampler_rng"), torch.Tensor)
            or not isinstance(saved.get("torch_rng"), torch.Tensor)
            or not isinstance(saved.get("train_seconds"), (int, float))
            or isinstance(saved.get("train_seconds"), bool)
            or not math.isfinite(saved["train_seconds"])
            or saved["train_seconds"] < 0
        ):
            raise ValueError("checkpoint lacks valid exact-resume state")
        expected_spec = WorldModel(
            variant, hidden or cfg.hidden, cfg.max_objects, bottleneck=bottleneck
        ).spec
        if saved["model_spec"] != expected_spec or saved["seed"] != seed:
            raise ValueError("resume model specification or seed differs")
        expected = asdict(cfg)
        previous = dict(saved["experiment"])
        expected.pop("steps")
        previous.pop("steps")
        if expected != previous:
            raise ValueError("resume experiment differs outside steps")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        optimizer.load_state_dict(saved["optimizer"])
        sampler.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        first_step, best = saved["completed_steps"], saved["best_validation"]
        elapsed_before = saved["train_seconds"]
        if first_step >= cfg.steps:
            raise ValueError("checkpoint is already complete; increase steps")
        # Keep the selected checkpoint when continuing into a fresh directory.
        best_source = Path(resume).parent / "best.pt"
        if output.resolve() != Path(resume).parent.resolve() and best_source.is_file():
            import shutil

            shutil.copy2(best_source, output / "best.pt")
    physics = Physics(global_coupling=cfg.global_coupling)
    train_data, train_dataset_record = load_dataset(
        DatasetSpec(
            "train",
            cfg.train_scenes,
            cfg.train_horizon,
            cfg.data_seed,
            cfg.max_objects,
            physics,
        )
    )
    validation, validation_dataset_record = load_dataset(
        DatasetSpec(
            "validation",
            cfg.eval_scenes,
            cfg.rollout_steps,
            cfg.data_seed,
            cfg.max_objects,
            physics,
        )
    )
    write_json(
        output / "config.json",
        {
            "experiment": asdict(cfg),
            "model_spec": model.spec,
            "seed": seed,
            "datasets": {
                "train": train_dataset_record,
                "validation": validation_dataset_record,
            },
        },
    )
    train_start = time.perf_counter()
    for step in range(first_step, cfg.steps):
        model.train()
        states, actions, mask = windows(
            train_data, cfg.batch_size, cfg.rollout_steps, sampler
        )
        current, objectives, quantizer_losses = states[:, 0], [], []
        for t in range(cfg.rollout_steps):
            current = model(current, actions[:, t], mask)
            objectives.append(loss_on_states(current, states[:, t + 1], mask))
            quantizer_losses.append(model.aux_loss)
        objective = objectives[0] + 0.5 * torch.stack(objectives).mean()
        objective = (
            objective + cfg.quantizer_weight * torch.stack(quantizer_losses).mean()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite objective at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % cfg.validate_every == 0 or completed == cfg.steps:
            metrics = evaluate(model, validation)
            score = metrics["position_rmse"] + 0.25 * metrics["velocity_rmse"]
            improved = score < best
            best = min(best, score)
            record = {
                "step": completed,
                "train_loss": float(objective.detach()),
                "validation_score": score,
                "validation": metrics,
            }
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            payload = {
                "format": "ripii-world-v1",
                "model": model.state_dict(),
                "model_spec": model.spec,
                "optimizer": optimizer.state_dict(),
                "experiment": asdict(cfg),
                "seed": seed,
                "completed_steps": completed,
                "sampler_rng": sampler.get_state(),
                "torch_rng": torch.get_rng_state(),
                "best_validation": best,
                "metrics": record,
                "train_seconds": elapsed_before + time.perf_counter() - train_start,
            }
            atomic_save(output / "latest.pt", payload)
            if improved:
                atomic_save(output / "best.pt", payload)
    atomic_save(output / "final.pt", payload)
    return load_model(output / "best.pt")


class Persistence(torch.nn.Module):
    """Copy the current state without using velocity or control."""

    def forward(self, state, action, mask):
        del action
        return state * mask[..., None]


class Kinematic(torch.nn.Module):
    """Advance constant velocity while ignoring forces and contacts."""

    def forward(self, state, action, mask):
        del action
        result = state.clone()
        result[..., :2] += 0.05 * state[..., 2:4]
        return result * mask[..., None]


class ForceKinematic(torch.nn.Module):
    """Apply force/mass and constant-acceleration motion without contacts."""

    def __init__(self, dt: float = 0.05):
        super().__init__()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.dt = dt

    def forward(self, state, action, mask):
        result = state.clone()
        acceleration = action / state[..., 5:6].clamp_min(0.1)
        result[..., 2:4] = state[..., 2:4] + self.dt * acceleration
        result[..., :2] = (
            state[..., :2] + self.dt * state[..., 2:4] + 0.5 * self.dt**2 * acceleration
        )
        return result * mask[..., None]


ANALYTIC_BASELINES = {
    "persistence": Persistence,
    "constant_velocity": Kinematic,
    "force_kinematic": ForceKinematic,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def widths(cfg, variants, bottleneck):
    target = sum(
        p.numel()
        for p in WorldModel(
            "multiscale", cfg.hidden, cfg.max_objects, bottleneck=bottleneck
        ).parameters()
    )
    result = {}
    for variant in variants:
        candidates = []
        for hidden in range(8, cfg.hidden * 4 + 1, 4):
            count = sum(
                p.numel()
                for p in WorldModel(
                    variant, hidden, cfg.max_objects, bottleneck=bottleneck
                ).parameters()
            )
            candidates.append((abs(count - target), hidden, count))
        _, hidden, count = min(candidates)
        result[variant] = {
            "hidden": hidden,
            "parameters": count,
            "relative_error": abs(count - target) / target,
        }
    return result


CAPACITY_MATCH_TOLERANCE = 0.05


def _benchmark_impl(
    cfg: Experiment,
    output: Path,
    seeds=(3, 7, 11),
    variants=VARIANTS,
    bottlenecks=("continuous",),
):
    cfg.validate()
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds
        )
        or not variants
        or len(set(variants)) != len(variants)
        or any(variant not in VARIANTS for variant in variants)
        or not bottlenecks
        or len(set(bottlenecks)) != len(bottlenecks)
        or any(
            bottleneck not in {"continuous", "fsq", "vq"} for bottleneck in bottlenecks
        )
    ):
        raise ValueError(
            "seeds, variants, and bottlenecks must be valid, nonempty, and unique"
        )
    capacity = {b: widths(cfg, variants, b) for b in bottlenecks}
    mismatched = {
        f"{variant}_{bottleneck}": details["relative_error"]
        for bottleneck, by_variant in capacity.items()
        for variant, details in by_variant.items()
        if details["relative_error"] > CAPACITY_MATCH_TOLERANCE
    }
    if mismatched:
        raise ValueError(f"capacity matching exceeds 5% tolerance: {mismatched}")
    if output.exists():
        raise ValueError(f"refusing existing benchmark directory: {output}")
    source_root = Path(__file__).parents[2]
    source_paths = sorted((source_root / "ripii").rglob("*.py"))
    source_paths.extend(
        p
        for p in (source_root / "pyproject.toml", source_root / "uv.lock")
        if p.is_file()
    )
    source_payloads = {path: path.read_bytes() for path in source_paths}
    sources = {
        str(path.relative_to(source_root)): hashlib.sha256(payload).hexdigest()
        for path, payload in source_payloads.items()
    }
    physics = Physics(global_coupling=cfg.global_coupling)
    dataset_specs = {
        "train": DatasetSpec(
            "train",
            cfg.train_scenes,
            cfg.train_horizon,
            cfg.data_seed,
            cfg.max_objects,
            physics,
        ),
        "validation": DatasetSpec(
            "validation",
            cfg.eval_scenes,
            cfg.rollout_steps,
            cfg.data_seed,
            cfg.max_objects,
            physics,
        ),
        **{
            split: DatasetSpec(
                split,
                cfg.eval_scenes,
                cfg.test_horizon,
                cfg.data_seed,
                cfg.max_objects,
                physics,
            )
            for split in ("test", "more_objects", "composition", "fast")
        },
    }
    protocol = {
        "format": "ripii-world-protocol-v1",
        "study": "RIPII object-state world model",
        "status": "development_benchmark",
        "experiment": asdict(cfg),
        "seeds": list(seeds),
        "variants": list(variants),
        "bottlenecks": list(bottlenecks),
        "capacity": capacity,
        "capacity_match_tolerance": CAPACITY_MATCH_TOLERANCE,
        "source_sha256": sources,
        "physics": physics.as_dict(),
        "datasets": {
            split: spec.as_record() for split, spec in dataset_specs.items()
        },
        "selection": "minimum validation position RMSE + 0.25 * velocity RMSE",
        "training": "same scenes, updates, sampler seeds and fixed objective weights",
        "uncertainty": "initialization/minibatch seeds on one fixed dataset; not population inference",
        "test_policy": "test splits used only after all training and validation selection",
        "analytic_baselines": {
            "persistence": "copies the current state",
            "constant_velocity": "advances velocity with no forces or contacts",
            "force_kinematic": "uses force/mass with no contacts, walls, or drag",
        },
        "advancement_rule": "continuous multiscale beats continuous graph OOD mean position RMSE by >=5% on every seed, ID regression <=5% on every seed",
        "limits": [
            "known object states, no learned visual perception",
            "soft-contact 2D simulator",
            "fixed training budget is equal updates, not equal wall-clock or FLOPs",
            "parameter matching uses nearest width; actual errors are reported",
            "no claim of physical rotation equivariance or compute-adaptive execution",
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    protocol_record = ExperimentProtocol.from_mapping(protocol)
    protocol = protocol_record.as_dict()
    cfg = Experiment(**protocol["experiment"])
    seeds = tuple(protocol["seeds"])
    variants = tuple(protocol["variants"])
    bottlenecks = tuple(protocol["bottlenecks"])
    output.mkdir(parents=True)
    # Written before training. Checkpoints are selected using validation only.
    write_json(output / "protocol.json", protocol)
    tracker = RunTracker.create(
        output,
        run_kind="world_benchmark",
        protocol_sha256=sha256(output / "protocol.json"),
    )
    tracker.transition("running")
    for path, payload in source_payloads.items():
        relative = path.relative_to(source_root)
        snapshot = output / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(payload)
    jobs = []
    for bottleneck in bottlenecks:
        for variant in variants:
            for seed in seeds:
                directory = output / f"{variant}_{bottleneck}" / f"seed_{seed}"
                print(
                    f"Training {variant}/{bottleneck} seed={seed} steps={cfg.steps}",
                    flush=True,
                )
                _, checkpoint = train(
                    cfg,
                    directory,
                    variant,
                    seed,
                    bottleneck,
                    capacity[bottleneck][variant]["hidden"],
                )
                jobs.append(
                    (
                        variant,
                        bottleneck,
                        seed,
                        directory,
                        checkpoint["completed_steps"],
                    )
                )
    loaded_datasets = {
        split: load_dataset(dataset_specs[split])
        for split in ("test", "more_objects", "composition", "fast")
    }
    datasets = {split: loaded[0] for split, loaded in loaded_datasets.items()}
    dataset_registry = {
        "format": "ripii-dataset-registry-v1",
        "datasets": {split: loaded[1] for split, loaded in loaded_datasets.items()},
    }
    write_json(output / "datasets.json", dataset_registry)
    rows = []
    analytic_baselines = {
        name: {split: evaluate(factory(), data) for split, data in datasets.items()}
        for name, factory in ANALYTIC_BASELINES.items()
    }
    for variant, bottleneck, seed, directory, selected_step in jobs:
        model, checkpoint = load_model(directory / "best.pt")
        metrics = {split: evaluate(model, data) for split, data in datasets.items()}
        row = {
            "variant": variant,
            "bottleneck": bottleneck,
            "seed": seed,
            "selected_step": selected_step,
            "train_seconds": torch.load(directory / "final.pt", weights_only=True)[
                "train_seconds"
            ],
            "parameters": sum(p.numel() for p in model.parameters()),
            "metrics": metrics,
        }
        write_json(directory / "evaluation.json", row)
        rows.append(row)
    grouped = {}
    for variant in variants:
        for bottleneck in bottlenecks:
            group = [
                r
                for r in rows
                if r["variant"] == variant and r["bottleneck"] == bottleneck
            ]
            aggregate = {}
            for split in datasets:
                values = [r["metrics"][split]["position_rmse"] for r in group]
                mean = sum(values) / len(values)
                std = (
                    sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
                ) ** 0.5
                aggregate[split] = {
                    "position_rmse_mean": mean,
                    "position_rmse_sample_std": std,
                }
            grouped[f"{variant}_{bottleneck}"] = aggregate
    comparisons = []
    for seed in seeds:
        a = next(
            (
                r
                for r in rows
                if r["variant"] == "multiscale"
                and r["bottleneck"] == "continuous"
                and r["seed"] == seed
            ),
            None,
        )
        b = next(
            (
                r
                for r in rows
                if r["variant"] == "graph"
                and r["bottleneck"] == "continuous"
                and r["seed"] == seed
            ),
            None,
        )
        if a and b:
            ood = ["more_objects", "composition", "fast"]
            a_ood = sum(a["metrics"][s]["position_rmse"] for s in ood) / len(ood)
            b_ood = sum(b["metrics"][s]["position_rmse"] for s in ood) / len(ood)
            improvement = 1 - a_ood / max(b_ood, 1e-12)
            id_regression = (
                a["metrics"]["test"]["position_rmse"]
                / max(b["metrics"]["test"]["position_rmse"], 1e-12)
                - 1
            )
            comparisons.append(
                {
                    "seed": seed,
                    "ood_relative_improvement": improvement,
                    "id_relative_regression": id_regression,
                    "passes": improvement >= 0.05 and id_regression <= 0.05,
                }
            )
    paired_controls = {}
    for baseline_variant in variants:
        if baseline_variant == "multiscale":
            continue
        pairs = []
        for seed in seeds:
            candidate = next(
                (
                    row
                    for row in rows
                    if row["variant"] == "multiscale"
                    and row["bottleneck"] == "continuous"
                    and row["seed"] == seed
                ),
                None,
            )
            baseline = next(
                (
                    row
                    for row in rows
                    if row["variant"] == baseline_variant
                    and row["bottleneck"] == "continuous"
                    and row["seed"] == seed
                ),
                None,
            )
            if candidate is None or baseline is None:
                continue
            ood = ("more_objects", "composition", "fast")
            candidate_ood = sum(
                candidate["metrics"][split]["position_rmse"] for split in ood
            ) / len(ood)
            baseline_ood = sum(
                baseline["metrics"][split]["position_rmse"] for split in ood
            ) / len(ood)
            ood_improvement = 1 - candidate_ood / max(baseline_ood, 1e-12)
            id_improvement = 1 - candidate["metrics"]["test"]["position_rmse"] / max(
                baseline["metrics"]["test"]["position_rmse"], 1e-12
            )
            pairs.append(
                {
                    "seed": seed,
                    "ood_relative_improvement": ood_improvement,
                    "id_relative_improvement": id_improvement,
                    "passes_5pct_ood_with_at_most_5pct_id_regression": (
                        ood_improvement >= 0.05 and id_improvement >= -0.05
                    ),
                }
            )
        if pairs:
            ood_values = [pair["ood_relative_improvement"] for pair in pairs]
            id_values = [pair["id_relative_improvement"] for pair in pairs]
            paired_controls[baseline_variant] = {
                "pairs": pairs,
                "ood_relative_improvement_mean": sum(ood_values) / len(ood_values),
                "ood_relative_improvement_sample_std": (
                    sum(
                        (value - sum(ood_values) / len(ood_values)) ** 2
                        for value in ood_values
                    )
                    / max(1, len(ood_values) - 1)
                )
                ** 0.5,
                "id_relative_improvement_mean": sum(id_values) / len(id_values),
                "ood_seed_wins": sum(value > 0 for value in ood_values),
                "seeds": len(pairs),
                "ood_relative_improvement_summary": descriptive_summary(ood_values),
                "id_relative_improvement_summary": descriptive_summary(id_values),
                "ood_exact_paired_sign_flip": paired_sign_flip_test(
                    ood_values, [0.0] * len(ood_values)
                ),
                "all_pass_5pct_gate": all(
                    pair["passes_5pct_ood_with_at_most_5pct_id_regression"]
                    for pair in pairs
                ),
                "claim_boundary": (
                    "Descriptive paired initialization-seed variation on one fixed "
                    "dataset; no population-level significance claim."
                ),
            }
    status = (
        "advance"
        if len(comparisons) >= 3 and all(r["passes"] for r in comparisons)
        else "no_advance"
    )
    report = {
        "protocol_sha256": sha256(output / "protocol.json"),
        "dataset_registry_sha256": sha256(output / "datasets.json"),
        "runs": rows,
        "by_model": grouped,
        "analytic_baselines": analytic_baselines,
        # Backwards-compatible alias for older report consumers.
        "kinematic_baseline": analytic_baselines["constant_velocity"],
        "paired_comparisons": comparisons,
        "paired_controls": paired_controls,
        "decision": status,
        "evidence_status": "development_only",
    }
    write_json(output / "summary.json", report)
    lines = [
        "# RIPII world-model benchmark",
        "",
        f"Decision: **{status}**. Development evidence only.",
        "",
        "Checkpoint selection used validation only. Values are position RMSE mean ± sample standard deviation across initialization seeds.",
        "",
        "| Model | In distribution | More objects | Held-out properties | Faster motion |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in grouped.items():
        cells = [
            f"{values[s]['position_rmse_mean']:.4f} ± {values[s]['position_rmse_sample_std']:.4f}"
            for s in datasets
        ]
        lines.append("| " + " | ".join([name] + cells) + " |")
    lines.extend(
        [
            "",
            "## Analytic baselines",
            "",
            "| Baseline | In distribution | More objects | Held-out properties | Faster motion |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in analytic_baselines.items():
        cells = [f"{values[split]['position_rmse']:.4f}" for split in datasets]
        lines.append("| " + " | ".join([name] + cells) + " |")
    lines.extend(
        [
            "",
            "## Advancement checks",
            "",
            json.dumps(comparisons, indent=2),
            "",
            "Three or more seeds must all meet the predeclared improvement and regression bounds. A pass remains a local simulator result, not external validation.",
            "",
            "## Paired control diagnostics",
            "",
            json.dumps(paired_controls, indent=2),
            "",
            "These control comparisons are descriptive fixed-dataset diagnostics, not population-level inference.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    tracker.transition("complete")
    artifacts = [
        {
            "path": str(p.relative_to(output)),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    write_json(
        output / "manifest.json",
        {"format": "ripii-world-manifest-v1", "artifacts": artifacts},
    )
    return report


def benchmark(
    cfg: Experiment,
    output: Path,
    seeds=(3, 7, 11),
    variants=VARIANTS,
    bottlenecks=("continuous",),
):
    """Run a complete benchmark and leave an explicit failed state on exceptions."""
    output = Path(output)
    status_existed = (output / "status.json").exists()
    try:
        return _benchmark_impl(cfg, output, seeds, variants, bottlenecks)
    except BaseException as exc:
        status = output / "status.json"
        if not status_existed and status.is_file() and not status.is_symlink():
            tracker = RunTracker.open(output)
            if tracker.payload["state"] != "failed":
                tracker.transition("failed", error=exc)
        raise


def verify(directory: Path):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be a regular non-symlink file")
    manifest_path = manifest_path.resolve()
    manifest = _load_json_object(manifest_path, "manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "ripii-world-manifest-v1"
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ValueError("invalid world-model manifest schema")
    root = directory.resolve()
    failures, seen = [], set()
    for entry in manifest["artifacts"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] < 0
        ):
            failures.append("<invalid manifest entry>")
            continue
        relative = entry["path"]
        raw = root / relative
        path = raw.resolve()
        if (
            relative in seen
            or root not in path.parents
            or raw.is_symlink()
            or not path.is_file()
        ):
            failures.append(relative)
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            failures.append(relative)
        elif path.suffix == ".json":
            try:
                _load_json_object(path, f"artifact {relative}")
            except ValueError:
                failures.append(relative)
        seen.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path != manifest_path and (path.is_file() or path.is_symlink())
    }
    unexpected = sorted(actual - seen)
    missing = sorted(seen - actual)
    if not seen or failures or unexpected or missing:
        raise ValueError(
            "artifact verification failed: "
            f"invalid={failures}, unexpected={unexpected}, missing={missing}"
        )
    protocol_path = root / "protocol.json"
    status_path = root / "status.json"
    if status_path.is_file():
        verify_complete_status(status_path, sha256(protocol_path))
    return {
        "status": "PASS",
        "artifacts_verified": len(seen),
        "unexpected_files": 0,
    }


def _capsule_signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "result_sha256"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture(directory: Path, output: Path):
    """Retain a compact, self-checksummed capsule after verifying a complete run."""
    directory, output = Path(directory), Path(output)
    verification = verify(directory)
    retained = {}
    for name in ("protocol.json", "summary.json", "report.md", "manifest.json"):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required result artifact is missing: {name}")
        content_text = path.read_text(encoding="utf-8")
        retained[name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "content_text": content_text,
        }
    summary = _parse_json_object(
        retained["summary.json"]["content_text"], "summary"
    )
    if summary.get("protocol_sha256") != retained["protocol.json"]["sha256"]:
        raise ValueError("summary does not reference the retained protocol")
    payload = {
        "format": "ripii-world-result-capsule-v2",
        "source_run": directory.name,
        "verification": verification,
        "decision": summary.get("decision"),
        "evidence_status": summary.get("evidence_status"),
        "retained": retained,
        "claim_boundary": (
            "This compact capsule retains the verified protocol, summary, report, "
            "and full-run manifest. Re-evaluation still requires the checkpoints "
            "listed by that manifest. Development results are not external validation."
        ),
    }
    payload["result_sha256"] = _capsule_signature(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("capsule output cannot be a symlink")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing existing capsule output: {output}") from exc
    return payload


def verify_capsule(path: Path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("capsule must be a regular non-symlink JSON file")
    payload = _load_json_object(path, "capsule")
    if (
        not isinstance(payload, dict)
        or payload.get("format")
        not in {"ripii-world-result-capsule-v1", "ripii-world-result-capsule-v2"}
        or not isinstance(payload.get("retained"), dict)
        or set(payload["retained"])
        != {"protocol.json", "summary.json", "report.md", "manifest.json"}
    ):
        raise ValueError("invalid world-model capsule schema")
    if payload.get("result_sha256") != _capsule_signature(payload):
        raise ValueError("world-model capsule signature mismatch")
    capsule_format = payload["format"]
    protocol = payload["retained"]["protocol.json"]
    summary = payload["retained"]["summary.json"]
    for name, entry in payload["retained"].items():
        content_key = (
            "content_text"
            if capsule_format == "ripii-world-result-capsule-v2"
            else "content"
        )
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("sha256"), str)
            or not isinstance(entry.get("bytes"), int)
            or content_key not in entry
        ):
            raise ValueError(f"invalid retained capsule entry: {name}")
        if capsule_format == "ripii-world-result-capsule-v2":
            content = entry["content_text"]
            if not isinstance(content, str):
                raise ValueError(f"retained text content is invalid: {name}")
            encoded = content.encode()
            if (
                len(encoded) != entry["bytes"]
                or hashlib.sha256(encoded).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"retained content hash or size mismatch: {name}")
            if name.endswith(".json"):
                _parse_json_object(content, f"retained JSON content {name}")
        elif not _finite_json(entry["content"]):
            raise ValueError(f"legacy retained content is non-finite: {name}")
    summary_content = (
        _parse_json_object(summary["content_text"], "retained summary")
        if capsule_format == "ripii-world-result-capsule-v2"
        else summary["content"]
    )
    if summary_content.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("capsule summary/protocol linkage is invalid")
    if payload.get("decision") != summary_content.get("decision"):
        raise ValueError("capsule decision does not match retained summary")
    return {
        "status": (
            "PASS"
            if capsule_format == "ripii-world-result-capsule-v2"
            else "LEGACY_SIGNATURE_ONLY"
        ),
        "result_sha256": payload["result_sha256"],
        "retained_files": len(payload["retained"]),
        "full_run_artifacts_verified_before_capture": payload["verification"].get(
            "artifacts_verified"
        ),
        "format": capsule_format,
        "content_hashes_verified": capsule_format == "ripii-world-result-capsule-v2",
    }
