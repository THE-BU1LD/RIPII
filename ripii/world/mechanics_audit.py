"""Deterministic MR representability checks; no training or model-selection study.

The lower bounds follow from the implemented scene momentum and semi-implicit
position constraints. They are not estimates of the missing historical matrix.
Run from a Git source checkout so the protocol can retain code and lockfile identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .experiment import loss_on_states
from .models import WorldModel, rollout
from .physics import Physics, simulate
from .run_status import RunTracker, verify_complete_status

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("ripii_mr", "ripii_mr_v2")
SEED = 1907
HORIZON = 8
TOLERANCE = 1e-9


def conservation_bounds(states, actions, mask, dt: float) -> dict[str, torch.Tensor]:
    """Per-coordinate 2D RMSE bounds under the stated P/Q constraints.

    For mass-weighted position Q and momentum P, MR fixes P'=P+dt*sum(F)
    and Q'=Q+dt*P'. Cauchy--Schwarz bounds the squared coordinate error
    from each scene's discrepancy in either conserved sum. The aggregate
    denominator matches world.evaluate: 2 * timesteps * live object count.
    Applying this to MR requires verifying those constraints: finite-precision
    mass floors and degenerate coarsening prevent a universal all-scene proof.
    """
    if states.ndim != 4 or states.shape[-1] != 6 or states.shape[1] < 2:
        raise ValueError("states must be [batch, time+1, objects, 6], time >= 1")
    batch, frames, objects, _ = states.shape
    if batch < 1 or objects < 1:
        raise ValueError("empty batch or object axis")
    if actions.shape != (batch, frames - 1, objects, 2):
        raise ValueError("actions must align with state transitions")
    if mask.shape != (batch, objects) or mask.dtype != torch.bool:
        raise ValueError("invalid Boolean scene mask")
    if not states.is_floating_point() or not actions.is_floating_point():
        raise TypeError("states and actions must be floating point")
    if states.dtype != actions.dtype or states.device != actions.device:
        raise ValueError("states and actions must share dtype and device")
    if mask.device != states.device or not mask.any(1).all():
        raise ValueError("mask device mismatch or empty scene")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (float, int))
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("dt must be finite and positive")
    if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise FloatingPointError("non-finite mechanics input")
    live = mask[:, None, :, None]
    if (states.masked_select(~live) != 0).any() or (
        actions.masked_select(~live) != 0
    ).any():
        raise ValueError("mechanics inputs require zero padding")
    properties = states[:, :1, :, 4:]
    if not torch.equal(states[..., 4:], properties.expand_as(states[..., 4:])):
        raise ValueError("radius and mass must remain constant")
    mass = states[:, 0, :, 5:6]
    if (mass[mask] < 1e-8).any() or (states[:, 0, :, 4][mask] <= 0).any():
        raise ValueError("positive radii and masses >= MR denominator floor required")
    # Accumulate the analytic constraints in float64 even for float32 trajectories.
    truth, forces, mass = states.double(), actions.double(), mass.double()
    momentum = (truth[:, 0, :, 2:4] * mass).sum(1)
    position = (truth[:, 0, :, :2] * mass).sum(1)
    momenta, positions = [], []
    for force in forces.unbind(1):
        momentum = momentum + dt * force.sum(1)
        position = position + dt * momentum
        momenta.append(momentum)
        positions.append(position)
    constrained_p = torch.stack(momenta, 1)
    constrained_q = torch.stack(positions, 1)
    target_p = (truth[:, 1:, :, 2:4] * mass[:, None]).sum(2)
    target_q = (truth[:, 1:, :, :2] * mass[:, None]).sum(2)
    mass_squared = mass.square().sum((1, 2))[:, None]
    count = mask.sum(1)[:, None]
    p_min_sse = (constrained_p - target_p).square().sum(-1) / mass_squared
    q_min_sse = (constrained_q - target_q).square().sum(-1) / mass_squared
    denominator = 2 * (frames - 1) * mask.sum()
    return {
        "constrained_momentum": constrained_p,
        "constrained_mass_weighted_position": constrained_q,
        "target_momentum": target_p,
        "target_mass_weighted_position": target_q,
        "velocity_rmse_lower_bound_by_scene_step": (p_min_sse / (2 * count)).sqrt(),
        "position_rmse_lower_bound_by_scene_step": (q_min_sse / (2 * count)).sqrt(),
        "velocity_rmse_lower_bound": (p_min_sse.sum() / denominator).sqrt(),
        "position_rmse_lower_bound": (q_min_sse.sum() / denominator).sqrt(),
    }


def fixture_specs() -> list[dict[str, Any]]:
    free = Physics(stiffness=0, damping=0, drag=0)
    drift = [0, 0, 0.4, 0, 0.07, 0.7]
    forced = [0, 0, 0, 0, 0.07, 0.7]

    def fixture(name, state, physics, action=None, reference=None, obstructed=False):
        return {
            "name": name,
            "initial_live_states": state,
            "constant_live_actions": action or [[0, 0] for _ in state],
            "physics": physics.as_dict(),
            "analytic_reference": reference,
            "expect_positive_position_bound": obstructed,
        }

    return [
        fixture("free_motion", [drift], free, reference="constant_force"),
        fixture(
            "drag_light",
            [drift],
            Physics(stiffness=0, damping=0),
            reference="linear_drag",
            obstructed=True,
        ),
        fixture(
            "drag_heavy",
            [[0, 0, 0.4, 0, 0.07, 1.4]],
            Physics(stiffness=0, damping=0),
            reference="linear_drag",
            obstructed=True,
        ),
        fixture(
            "wall_contact",
            [[0.95, 0, 0, 0, 0.1, 1]],
            Physics(drag=0),
            reference="wall_first_step",
            obstructed=True,
        ),
        fixture(
            "force_four_substeps", [forced], free, [[0.3, 0]], "constant_force", True
        ),
        fixture(
            "force_one_substep",
            [forced],
            Physics(substeps=1, stiffness=0, damping=0, drag=0),
            [[0.3, 0]],
            "constant_force",
        ),
        fixture(
            "internal_contact",
            [[-0.08, 0, 0, 0, 0.1, 1], [0.08, 0, 0, 0, 0.1, 1]],
            Physics(drag=0),
        ),
        fixture(
            "long_range_coupling",
            [[-0.65, 0, 0, 0, 0.07, 1], [0.65, 0, 0, 0, 0.07, 1]],
            Physics(stiffness=0, damping=0, drag=0, global_coupling=1),
        ),
        fixture(
            "five_objects_drag",
            [
                [x, 0, 0.2, 0, 0.07, 0.7 if i % 2 == 0 else 1.4]
                for i, x in enumerate([-0.6, -0.3, 0, 0.3, 0.6])
            ],
            Physics(stiffness=0, damping=0),
            reference="linear_drag",
            obstructed=True,
        ),
    ]


def fixture_tensors(spec):
    count = len(spec["initial_live_states"])
    state = torch.zeros(1, 8, 6, dtype=torch.float64)
    state[0, :count] = torch.tensor(spec["initial_live_states"], dtype=torch.float64)
    mask = torch.arange(8)[None] < count
    actions = torch.zeros(1, HORIZON, 8, 2, dtype=torch.float64)
    actions[0, :, :count] = torch.tensor(
        spec["constant_live_actions"], dtype=torch.float64
    )
    return state, actions, mask


def analytic_error(spec, states) -> float | None:
    """Independent closed forms where the fixture stays in a single force regime."""
    kind = spec["analytic_reference"]
    if kind is None:
        return None
    cfg = Physics(**spec["physics"])
    count = len(spec["initial_live_states"])
    start = states[0, 0, :count]
    initial_x, initial_v, mass = start[:, :2], start[:, 2:4], start[:, 5:6]
    step = torch.arange(1, HORIZON + 1, dtype=torch.float64)[:, None, None]
    if kind == "constant_force":
        acceleration = (
            torch.tensor(spec["constant_live_actions"], dtype=torch.float64) / mass
        )
        velocity = initial_v + step * cfg.dt * acceleration
        coefficient = step * (step - 1) / 2 + step * (cfg.substeps + 1) / (
            2 * cfg.substeps
        )
        position = (
            initial_x
            + step * cfg.dt * initial_v
            + coefficient * cfg.dt**2 * acceleration
        )
    elif kind == "linear_drag":
        h = cfg.dt / cfg.substeps
        q = 1 - h * cfg.drag / mass
        power = q ** (step * cfg.substeps)
        velocity = power * initial_v
        position = initial_x + h * initial_v * q * (1 - power) / (1 - q)
    elif kind == "wall_first_step":
        # The first observation stays in positive right-wall penetration with
        # an unsaturated inward spring/damping force. Matrix powers solve that
        # linear recurrence independently of simulate's contact implementation.
        h, m = cfg.dt / cfg.substeps, float(mass[0, 0])
        k, c = cfg.stiffness, cfg.damping
        transition = torch.tensor(
            [[1 - h * h * k / m, h * (1 - h * c / m)], [-h * k / m, 1 - h * c / m]],
            dtype=torch.float64,
        )
        right_wall = 1 - start[0, 4]
        z = torch.stack([initial_x[0, 0] - right_wall, initial_v[0, 0]])
        for _ in range(cfg.substeps):
            if z[0] <= 0 or k * z[0] + c * z[1] <= 0:
                raise ValueError("wall reference left its declared linear regime")
            z = transition @ z
        expected = torch.stack([z[0] + right_wall, z[1]])
        return float((states[0, 1, 0, [0, 2]] - expected).abs().max())
    else:
        raise ValueError("unknown analytic fixture reference")
    expected = torch.cat([position, velocity], -1)
    return float((states[0, 1:, :count, :4] - expected).abs().max())


def diagnostic_model(variant):
    torch.manual_seed(SEED)
    model = WorldModel(variant, hidden=16).double()
    with torch.no_grad():
        for layer in (
            model.multiresolution.fine_impulse.net[-1],
            model.multiresolution.coarse_impulse.net[-1],
        ):
            assert isinstance(layer, nn.Linear)
            assert layer.bias is not None
            layer.weight.normal_(0, 0.005)
            layer.bias.fill_(0.02)
        router_output = model.multiresolution.router[-1]
        assert isinstance(router_output, nn.Linear)
        assert router_output.bias is not None
        router_output.weight.zero_()
        router_output.bias.fill_(1)
    return model


def routing_probe() -> list[dict[str, Any]]:
    """Count real coarse executions and measure task gradients, not speedup."""
    spec = next(row for row in fixture_specs() if row["name"] == "internal_contact")
    state, actions, mask = fixture_tensors(spec)
    rows = []
    for variant in VARIANTS:
        for on in (False, True):
            model = diagnostic_model(variant).train()
            with torch.no_grad():
                model.multiresolution.router[-1].bias.fill_(1 if on else -1)
            calls: list[int] = []
            model.multiresolution.coarsener.register_forward_hook(
                lambda _m, args, _out, calls=calls: calls.append(args[0].shape[0])
            )
            prediction = model(state, actions[:, 0], mask)
            target = simulate(state, actions[:, 0], mask, Physics(**spec["physics"]))
            loss_on_states(prediction, target, mask).backward()
            gradient = sum(
                float(p.grad.abs().sum())
                for p in model.multiresolution.router.parameters()
                if p.grad is not None
            )
            rows.append(
                {
                    "variant": variant,
                    "route_on": on,
                    "coarsener_calls": len(calls),
                    "coarse_scenes_processed": sum(calls),
                    "router_task_gradient_l1": gradient,
                    "routing_probability": model.diagnostics(mask)[
                        "routing_probability_mean"
                    ],
                }
            )
    return rows


def _bytes(payload) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(result) -> str:
    lines = [
        "# RIPII-MR deterministic mechanics audit",
        "",
        "No training, checkpoint selection, scientific advancement decision, or historical matrix replay was performed. Values below are lower bounds on per-coordinate RMSE over eight steps, conditional on the conservative P/Q constraints verified on every retained initialized rollout. The single-object obstructions are independent of learned weights because there are no internal pair impulses. A zero bound does not establish accurate trajectories.",
        "",
        "| Fixture | Position RMSE lower bound | Velocity RMSE lower bound | Closed-form reference max error |",
        "|---|---:|---:|---:|",
    ]
    for row in result["fixtures"]:
        error = row["analytic_reference_max_error"]
        reference = "not applicable" if error is None else f"{error:.3e}"
        lines.append(
            f"| {row['name']} | {row['bounds']['position_rmse_lower_bound']:.9g} | {row['bounds']['velocity_rmse_lower_bound']:.9g} | {reference} |"
        )
    lines += [
        "",
        "The wall-contact analytic reference covers the first observation only. Free-motion, drag, and constant-force closed forms cover all eight observations. Contact and coupling controls are checked through their conserved sums. The multi-object bounds are conditional on actual conservation: mass floors and degenerate or tied coarse assignments prevent a universal all-scene guarantee.",
    ]
    lines += [
        "",
        "Both variants impose P'=P+dt*sum(actions) and Q'=Q+dt*P', where Q is mass-weighted position. Simulator wall and drag forces are additional external impulses. Four simulator substeps also differ from MR's single semi-implicit position update under constant force. Internal contact and symmetric coupling are negative controls because their net forces cancel.",
        "",
        "## Actual coarse execution and router task gradient",
        "",
        "| Variant | Route on | Coarsener calls | Scenes processed | Router gradient L1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["routing_probe"]:
        lines.append(
            f"| {row['variant']} | {row['route_on']} | {row['coarsener_calls']} | {row['coarse_scenes_processed']} | {row['router_task_gradient_l1']:.9g} |"
        )
    lines += [
        "",
        "These counters measure execution, not an accuracy/latency benefit. The v2 skipped path has no task gradient to the router for that scene; the logits in this probe are ±1, not saturated. The dynamics equations and checkpoint schema were not changed. The diagnostic uses the disclosed untrained parameter interventions. The historical 75-run result remains source-reported because its full artifacts are absent from the checkout.",
        "",
        "All fixture inputs, simulator targets, initialized model predictions, physics settings, weight files, source snapshots and integrity hashes are retained alongside this report. No claim apportions the historical performance gap to this obstruction.",
        "",
    ]
    return "\n".join(lines)


def run_audit(output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    source = sorted(ROOT.glob("ripii/**/*.py")) + [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    protocol: dict[str, Any] = {
        "format": "ripii-mr-mechanics-audit-v1",
        "hypothesis": "MR's action-only total momentum and single-step integration conflict with drag, walls, and substepped constant-force targets; internal-force-only controls have zero bound.",
        "evidence_kind": "deterministic synthetic construction diagnostic",
        "independent_unit": "explicit fixture; no population inference or seed significance",
        "data_identity": "literal fixture states/actions and Physics parameters below",
        "splits_and_selection": "no train/test split, holdout access, training, tuning, or selection",
        "variants": list(VARIANTS),
        "routing_probe": {
            "fixture": "internal_contact",
            "logits": [-1, 1],
            "loss": "loss_on_states(prediction, simulator_next_state, mask): position weight4, velocity weight1",
            "optimizer_updates": 0,
        },
        "initialization_seed": SEED,
        "weight_intervention": "fine/coarse final weights N(0,0.005), bias0.02; router output weight0,bias1; no fitted weights",
        "primary_measurement": "Cauchy lower bounds on aggregate per-coordinate position/velocity RMSE excluding the initial frame",
        "uncertainty": "analytic constraint plus measured floating-point residual; no statistical confidence interval",
        "budget": {
            "cpu_lanes": 1,
            "torch_threads": 1,
            "max_seconds": 120,
            "training_updates": 0,
            "horizon": HORIZON,
            "hidden": 16,
            "max_objects": 8,
        },
        "checks": {
            "absolute_tolerance": TOLERANCE,
            "positive_bound_minimum": 1e-8,
            "failure_rule": "any nonfinite value, failed reference, violated bound/constraint, or missing fixture fails the audit; preserve failed output",
        },
        "fixtures": fixture_specs(),
        "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in source},
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_bytes(_bytes(protocol))
    tracker = RunTracker.create(
        output,
        run_kind="deterministic-mechanics-audit",
        protocol_sha256=_hash(protocol_path),
    )
    tracker.transition("running")
    start = time.perf_counter()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_rng = torch.random.get_rng_state()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for path in source:
            destination = output / "source_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        environment = {
            "python": sys.version,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float64",
            "threads": torch.get_num_threads(),
            "command": sys.argv,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_dirty_files": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).splitlines(),
        }
        (output / "environment.json").write_bytes(_bytes(environment))
        models = {variant: diagnostic_model(variant).eval() for variant in VARIANTS}
        for variant, model in models.items():
            torch.save(model.state_dict(), output / f"{variant}_untrained_weights.pt")
        rows = []
        for spec in protocol["fixtures"]:
            state, actions, mask = fixture_tensors(spec)
            physics = Physics(**spec["physics"])
            trajectory = [state]
            for action in actions.unbind(1):
                trajectory.append(simulate(trajectory[-1], action, mask, physics))
            truth = torch.stack(trajectory, 1)
            bounds = conservation_bounds(truth, actions, mask, physics.dt)
            reference_error = analytic_error(spec, truth)
            if reference_error is not None and reference_error > TOLERANCE:
                raise AssertionError(f"analytic reference failed: {spec['name']}")
            position_bound = float(bounds["position_rmse_lower_bound"])
            if spec["expect_positive_position_bound"]:
                if position_bound <= 1e-8:
                    raise AssertionError(f"expected obstruction absent: {spec['name']}")
            elif position_bound > TOLERANCE:
                raise AssertionError(f"negative control failed: {spec['name']}")
            model_rows = {}
            for variant, model in models.items():
                with torch.no_grad():
                    prediction = rollout(model, state, actions, mask)
                mass = state[..., 5:6]
                momentum = (prediction[:, 1:, :, 2:4] * mass[:, None]).sum(2)
                position = (prediction[:, 1:, :, :2] * mass[:, None]).sum(2)
                p_residual = float(
                    (momentum - bounds["constrained_momentum"]).abs().max()
                )
                q_residual = float(
                    (position - bounds["constrained_mass_weighted_position"])
                    .abs()
                    .max()
                )
                error = (prediction[:, 1:] - truth[:, 1:]).square() * mask[
                    :, None, :, None
                ]
                denominator = 2 * HORIZON * mask.sum()
                p_rmse = float((error[..., 2:4].sum() / denominator).sqrt())
                q_rmse = float((error[..., :2].sum() / denominator).sqrt())
                if max(p_residual, q_residual) > TOLERANCE:
                    raise AssertionError(
                        f"MR conservation constraint failed: {variant}/{spec['name']}"
                    )
                if q_rmse + TOLERANCE < position_bound or p_rmse + TOLERANCE < float(
                    bounds["velocity_rmse_lower_bound"]
                ):
                    raise AssertionError(
                        f"lower bound violated: {variant}/{spec['name']}"
                    )
                model_rows[variant] = {
                    "untrained_prediction": prediction.tolist(),
                    "position_rmse": q_rmse,
                    "velocity_rmse": p_rmse,
                    "momentum_constraint_residual_max": p_residual,
                    "position_constraint_residual_max": q_residual,
                    "diagnostics_last_step": model.diagnostics(mask),
                }
            rows.append(
                {
                    "name": spec["name"],
                    "analytic_reference_max_error": reference_error,
                    "truth": truth.tolist(),
                    "bounds": {name: value.tolist() for name, value in bounds.items()},
                    "models": model_rows,
                }
            )
            # Preserve already completed fixtures if a later fixture fails.
            (output / "measurements.json").write_bytes(
                _bytes(
                    {
                        "format": "ripii-mr-mechanics-measurements-v1",
                        "completed_fixture_count": len(rows),
                        "training_updates": 0,
                        "fixtures": rows,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
            )
            if time.perf_counter() - start > 120:
                raise TimeoutError("mechanics audit exceeded its 120-second budget")
        routes = routing_probe()
        if time.perf_counter() - start > 120:
            raise TimeoutError("mechanics audit exceeded its 120-second budget")
        for row in routes:
            expected_calls = (
                0 if row["variant"] == "ripii_mr_v2" and not row["route_on"] else 1
            )
            if row["coarsener_calls"] != expected_calls:
                raise AssertionError(
                    "routing execution count disagrees with source semantics"
                )
        result = {
            "format": "ripii-mr-mechanics-measurements-v1",
            "completed_fixture_count": len(rows),
            "training_updates": 0,
            "fixtures": rows,
            "routing_probe": routes,
            "elapsed_seconds": time.perf_counter() - start,
        }
        (output / "measurements.json").write_bytes(_bytes(result))
        (output / "report.md").write_text(_report(result), encoding="utf-8")
        tracker.transition("complete")
        files = sorted(path for path in output.rglob("*") if path.is_file())
        manifest = {
            "format": "ripii-mr-mechanics-manifest-v1",
            "artifacts": [
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _hash(path),
                }
                for path in files
            ],
        }
        (output / "manifest.json").write_bytes(_bytes(manifest))
        return verify_audit(output)
    except BaseException as exc:
        tracker.transition("failed", error=exc)
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        torch.random.set_rng_state(previous_rng)


def verify_audit(output: Path) -> dict[str, Any]:
    """Verify this diagnostic's complete file set and source snapshot integrity.

    This is integrity/completion checking, not independent scientific reproduction
    or an authenticated external digest anchor.
    """
    if output.is_symlink() or not output.is_dir():
        raise ValueError("audit must be a regular directory")
    paths = list(output.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("audit must not contain symlinks")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("format") != "ripii-mr-mechanics-manifest-v1":
        raise ValueError("unknown mechanics manifest")
    seen = set()
    for entry in manifest["artifacts"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("unsafe or duplicate artifact path")
        seen.add(str(relative))
        path = output / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _hash(path) != entry["sha256"]
        ):
            raise ValueError(f"artifact integrity failed: {relative}")
    actual = {str(path.relative_to(output)) for path in paths if path.is_file()} - {
        "manifest.json"
    }
    if actual != seen:
        raise ValueError("artifact file set mismatch")
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("format") != "ripii-mr-mechanics-audit-v1"
        or protocol.get("fixtures") != fixture_specs()
        or protocol.get("variants") != list(VARIANTS)
        or protocol.get("initialization_seed") != SEED
        or protocol.get("budget", {}).get("horizon") != HORIZON
        or protocol.get("budget", {}).get("training_updates") != 0
    ):
        raise ValueError("incompatible or incomplete mechanics protocol")
    verify_complete_status(output / "status.json", _hash(protocol_path))
    required_source = {
        "ripii/world/experiment.py",
        "ripii/world/mechanics_audit.py",
        "ripii/world/models.py",
        "ripii/world/multiresolution.py",
        "ripii/world/physics.py",
        "ripii/world/run_status.py",
        "pyproject.toml",
        "uv.lock",
    }
    if not required_source.issubset(protocol.get("source_sha256", {})):
        raise ValueError("incomplete source provenance")
    for name, digest in protocol["source_sha256"].items():
        if (
            str(Path(name)) != name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("unsafe source path")
        if _hash(output / "source_snapshot" / name) != digest:
            raise ValueError("source snapshot mismatch")
    result = json.loads((output / "measurements.json").read_text())
    if (
        result.get("format") != "ripii-mr-mechanics-measurements-v1"
        or result.get("training_updates") != 0
        or [row["name"] for row in result["fixtures"]]
        != [row["name"] for row in protocol["fixtures"]]
        or result["completed_fixture_count"] != len(protocol["fixtures"])
        or [
            (row["variant"], row["route_on"]) for row in result.get("routing_probe", [])
        ]
        != [(variant, on) for variant in VARIANTS for on in (False, True)]
        or any(
            set(row.get("models", {})) != set(VARIANTS) for row in result["fixtures"]
        )
    ):
        raise ValueError("incomplete fixture record")
    return {
        "status": "verified_integrity_and_completion",
        "artifacts": len(seen),
        "fixtures": result["completed_fixture_count"],
        "manifest_sha256": _hash(output / "manifest.json"),
        "elapsed_seconds": result["elapsed_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output", type=Path, help="fresh output directory; never overwritten"
    )
    mode.add_argument(
        "--verify",
        type=Path,
        help="verify retained diagnostic integrity and completion",
    )
    args = parser.parse_args()
    result = verify_audit(args.verify) if args.verify else run_audit(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
