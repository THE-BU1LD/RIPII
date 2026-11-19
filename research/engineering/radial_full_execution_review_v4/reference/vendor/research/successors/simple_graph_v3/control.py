"""Standard-library integrity, resource admission, and fail-closed v3 analysis."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
V1 = ROOT.parent / "simple_graph_v1"
V2 = ROOT.parent / "simple_graph_v2"
CANDIDATE = "radial_mixture"
PREDECESSOR_BASELINE = "adaptive_context_v2"
PRIMARY_BASELINES = (PREDECESSOR_BASELINE, "graph", "simple_pool", "transformer")
GATE_ABLATION = "radial_fixed"
NUMERICAL_TEST_COUNT = 10
REQUIRED_RUNTIME_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
ENGINEERING_SPEC_OVERRIDES = {
    "data_seeds": [990037],
    "model_seeds": [7003],
    "steps": 2,
    "train_scenes": 8,
    "eval_scenes": 4,
    "forecast": 4,
    "batch_size": 4,
    "validate_every": 1,
    "curriculum": [[2, 2]],
    "max_seconds": 900,
}


def dataset_seed(domain, block, split, regime="standard"):
    """Role-separated deterministic seed; every dataset artifact gets one identity."""
    split_code = {"train": 1, "validation": 2, "test": 3}
    regime_code = {
        "standard": 0,
        "strong_interaction": 10,
        "fast": 20,
        "noisy": 30,
    }
    if domain not in {"springs", "charged"}:
        raise ValueError(f"Unknown domain {domain}")
    if split not in split_code or regime not in regime_code:
        raise ValueError("Unknown dataset split or regime")
    if split != "test" and regime != "standard":
        raise ValueError("Distribution shifts are test-only")
    domain_code = 0 if domain == "springs" else 50
    return block * 100 + domain_code + split_code[split] + regime_code[regime]


def simulator_parameters(domain, regime, objects=5):
    """Return the exact pinned simulator parameters for one declared regime."""
    if domain not in {"springs", "charged"}:
        raise ValueError(f"Unknown domain {domain}")
    if isinstance(objects, bool) or not isinstance(objects, int) or objects < 2:
        raise ValueError("Simulator object count must be an integer of at least two")
    parameters: dict[str, int | float] = {"n_balls": objects}
    if regime == "strong_interaction":
        parameters["interaction_strength"] = 0.15 if domain == "springs" else 1.5
    elif regime == "fast":
        parameters["vel_norm"] = 0.75
    elif regime == "noisy":
        parameters["noise_var"] = 0.01
    elif regime != "standard":
        raise ValueError(f"Unknown regime {regime}")
    return parameters


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")


def load_json(path):
    """Load strict JSON, rejecting non-finite constants and unreadable receipts."""

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON receipt: {path}") from exc


def _finite_number(value, label, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return float(value)


def accumulated_charge(execution_root):
    """Fail closed while summing every retained campaign charge receipt."""
    total = 0.0
    for path in sorted(Path(execution_root).glob("*/charge.json")):
        receipt = load_json(path)
        if not isinstance(receipt, dict):
            raise ValueError(f"Charge receipt must be an object: {path}")
        total += _finite_number(receipt.get("seconds"), f"{path}: seconds")
    if not math.isfinite(total):
        raise ValueError("Cumulative campaign charge is non-finite")
    return total


def validated_calibration(payload, expected_models, expected_training_horizon=None):
    """Validate one exact, finite full-shape timing row per frozen model."""
    if not isinstance(payload, list) or len(payload) != len(expected_models):
        raise ValueError("Calibration must contain exactly one row per model")
    indexed = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Calibration row {index} must be an object")
        model = row.get("model")
        if not isinstance(model, str) or model in indexed:
            raise ValueError("Calibration model names must be unique strings")
        _finite_number(
            row.get("step_seconds_max"),
            f"Calibration {model} step_seconds_max",
            positive=True,
        )
        _finite_number(
            row.get("validation_seconds"),
            f"Calibration {model} validation_seconds",
            positive=True,
        )
        for name in ("parameters", "hidden"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Calibration {model} {name} must be a positive integer")
        horizon = row.get("training_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(
                f"Calibration {model} training_horizon must be a positive integer"
            )
        if (
            expected_training_horizon is not None
            and horizon != expected_training_horizon
        ):
            raise ValueError(
                f"Calibration {model} was not measured at the frozen horizon"
            )
        indexed[model] = row
    if set(indexed) != set(expected_models):
        raise ValueError("Calibration model set differs from the frozen protocol")
    return [indexed[model] for model in expected_models]


def execution_spec(protocol, mode):
    """Construct the only admitted stage specification from the frozen protocol."""
    if not isinstance(protocol, dict):
        raise ValueError("Frozen protocol must be a JSON object")
    if mode not in {"test", "fixture", "development"}:
        raise ValueError(f"Unsupported execution mode: {mode}")
    spec = {**protocol, "mode": mode}
    if mode in {"test", "fixture"}:
        spec.update(ENGINEERING_SPEC_OVERRIDES)
    return spec


def validate_execution_spec(spec, protocol):
    """Reject header drift while permitting only a reduced remaining wall cap."""
    if not isinstance(spec, dict):
        raise ValueError("Run specification must be a JSON object")
    expected = execution_spec(protocol, spec.get("mode"))
    if set(spec) != set(expected):
        raise ValueError("Run specification keys differ from the admitted contract")
    for key, value in expected.items():
        if key != "max_seconds" and spec.get(key) != value:
            raise ValueError(f"Run specification drift: {key}")
    cap = _finite_number(spec.get("max_seconds"), "Run max_seconds", positive=True)
    if cap > expected["max_seconds"]:
        raise ValueError("Run wall cap exceeds the admitted contract")
    return spec


def predecessor():
    portable = ROOT / "PORTABLE.json"
    if portable.is_file():
        receipt = json.loads(portable.read_text(encoding="utf-8"))["origin_predecessor"]
        return {**receipt, "scope": "origin receipt embedded in portable source"}
    expected = json.loads((ROOT / "PREDECESSOR.json").read_text(encoding="utf-8"))
    changed = [
        name
        for name, digest in expected["frozen_paths"].items()
        if not (REPO / name).is_file() or sha(REPO / name) != digest
    ]
    state = json.loads((V2 / "STATE.json").read_text(encoding="utf-8"))
    semantic_ok = (
        state.get("status") == expected["required_v2_status"]
        and state.get("scientific_result")
        == expected["required_v2_scientific_result"]
        and state.get("advance") is expected["required_v2_advance"]
    )
    return {
        "status": "PASS" if not changed and semantic_ok else "FAIL",
        "files": len(expected["frozen_paths"]),
        "changed": changed,
        "semantic_ok": semantic_ok,
    }


def fingerprint():
    paths = list((REPO / "ripii").rglob("*.py"))
    paths += list(ROOT.glob("*.py"))
    paths += [ROOT / "protocol.json", ROOT / "PROTOCOL.md", ROOT / "PREDECESSOR.json"]
    paths += [V1 / "models.py", V2 / "models.py"]
    paths += [
        path
        for path in (V1 / "upstream").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def _package_identity(name):
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "version": None,
            "metadata_sha256": None,
            "wheel_sha256": None,
            "record_sha256": None,
        }
    files = distribution.files or []
    result = {"version": distribution.version}
    for filename, label in (
        ("METADATA", "metadata_sha256"),
        ("WHEEL", "wheel_sha256"),
        ("RECORD", "record_sha256"),
    ):
        matches = [item for item in files if str(item).endswith(f"/{filename}")]
        if len(matches) != 1:
            result[label] = None
            continue
        path = Path(distribution.locate_file(matches[0]))
        result[label] = (
            sha(path) if path.is_file() and not path.is_symlink() else None
        )
    return result


def identity():
    """Exact source/runtime identity required across every qualification stage."""
    package_records = {
        name: _package_identity(name) for name in ("numpy", "torch")
    }
    return {
        "sources": fingerprint(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: record["version"] for name, record in package_records.items()
        },
        "package_records": package_records,
        "runtime_policy": {
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "torch_threads": 1,
            "environment": REQUIRED_RUNTIME_ENV,
        },
    }


def validate_runtime_environment(environment=None):
    """Require deterministic process settings before numerical-library imports."""
    environment = os.environ if environment is None else environment
    mismatches = {
        name: environment.get(name)
        for name, expected in REQUIRED_RUNTIME_ENV.items()
        if environment.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Runtime environment differs from policy: {mismatches}")
    return {name: environment[name] for name in REQUIRED_RUNTIME_ENV}


def validate_run_path(path, *, allow_shard=False):
    """Confine workers to a direct execution attempt or its direct shard."""
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("Run path may not be a symlink")
    resolved = supplied.resolve()
    execution = (ROOT / "execution").resolve()
    try:
        relative = resolved.relative_to(execution)
    except ValueError as exc:
        raise ValueError("Run path is outside the execution root") from exc
    depths = {1, 3} if allow_shard else {1}
    if len(relative.parts) not in depths:
        raise ValueError("Run path has an unsupported execution depth")
    if len(relative.parts) == 3 and relative.parts[1] != "shards":
        raise ValueError("Worker sub-run is not a declared shard")
    if not resolved.is_dir():
        raise ValueError("Run path must be a directory")
    return resolved


def acquire_campaign_lock(path=None):
    """Acquire a crash-recoverable kernel lock while retaining an owner receipt."""
    path = Path(path) if path is not None else ROOT / "execution" / "active.lock"
    path.parent.mkdir(exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("Another v3 campaign owns the execution lock") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_campaign_lock(handle):
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def owned_process_ids(rows, own_pid, allowed_pids=()):
    """Return this runner's lineage/children without absorbing sibling processes."""
    parent_by_pid = {int(pid): int(parent) for pid, parent, _ in rows}
    owned = {own_pid, *allowed_pids}
    for _ in range(len(rows)):
        descendants = {
            int(pid) for pid, parent, _ in rows if int(parent) in owned
        }
        if descendants <= owned:
            break
        owned |= descendants
    cursor = own_pid
    for _ in range(len(rows)):
        parent = parent_by_pid.get(cursor)
        if parent is None or parent <= 1 or parent in owned:
            break
        owned.add(parent)
        cursor = parent
    return owned


def sample(allowed_pids=()):
    result: dict[str, Any] = {"time": time.time(), "errors": []}
    try:
        if sys.platform == "darwin":
            raw = subprocess.check_output(["vm_stat"], text=True)
            page_size = re.search(r"page size of (\d+)", raw)
            if page_size is None:
                raise RuntimeError("vm_stat output omitted page size")
            size = int(page_size.group(1))
            counters = {
                key.strip(): int(value)
                for key, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", raw, re.M)
            }
            result.update(
                free=(counters["Pages free"] + counters["Pages speculative"]) * size,
                pageouts=counters["Pageouts"],
                swapins=counters["Swapins"],
                swapouts=counters["Swapouts"],
            )
        elif sys.platform.startswith("linux"):
            mem = dict(
                re.findall(
                    r"^(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M
                )
            )
            vm = dict(
                line.split() for line in Path("/proc/vmstat").read_text().splitlines()
            )
            result.update(
                free=int(mem["MemAvailable"]) * 1024,
                free_metric="MemAvailable",
                pageouts=int(vm["pswpout"]),
                swapins=int(vm["pswpin"]),
                swapouts=int(vm["pswpout"]),
            )
        else:
            raise RuntimeError("Unsupported resource guard OS")
        processes = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,command="], text=True, stderr=subprocess.PIPE
        )
        rows = [line.split(None, 2) for line in processes.splitlines() if line.strip()]
        owned = owned_process_ids(rows, os.getpid(), allowed_pids)
        result["conflicts"] = [
            f"{pid} {command}"
            for pid, _, command in rows
            if "simple_graph_v3/" in command and int(pid) not in owned
        ]
    except Exception as exc:
        result["errors"].append(str(exc))
    result["disk"] = shutil.disk_usage(ROOT).free
    return result


def assess(samples):
    reasons = []
    if len(samples) != 2:
        reasons.append("TWO_SAMPLES_REQUIRED")
    for row in samples:
        reasons.extend(row.get("errors", []))
        if row.get("free", 0) < 512 * 1024**2:
            reasons.append("MEMORY_BELOW_512_MIB")
        if row.get("disk", 0) < 10 * 1024**3:
            reasons.append("DISK_BELOW_10_GIB")
        if row.get("conflicts"):
            reasons.append("CONFLICTING_RUNNER")
    if len(samples) == 2:
        for key in ("pageouts", "swapins", "swapouts"):
            if any(key not in row for row in samples):
                reasons.append(f"MISSING_{key}")
            elif samples[1][key] > samples[0][key]:
                reasons.append(f"INCREASING_{key}")
    return {
        "status": "BLOCKED" if reasons else "PASS",
        "reasons": sorted(set(reasons)),
        "samples": samples,
    }


def admission(allowed_pids=()):
    first = sample(allowed_pids)
    time.sleep(3)
    return assess([first, sample(allowed_pids)])


def summarize(rows, spec):
    expected = {
        (domain, block, seed, model)
        for domain in spec["domains"]
        for block in spec["data_seeds"]
        for seed in spec["model_seeds"]
        for model in spec["models"]
    }
    indexed = {}
    for row in rows:
        key = tuple(
            row[name] for name in ("domain", "data_seed", "model_seed", "model")
        )
        if key in indexed or key not in expected:
            raise ValueError("Duplicate or unexpected evaluation cell")
        regimes = row.get("regimes", {})
        if set(regimes) != set(spec["regimes"]):
            raise ValueError("Missing or unexpected evaluation regime")
        for metric in regimes.values():
            if (
                not math.isfinite(metric["position_rmse"])
                or metric["position_rmse"] < 0
            ):
                raise ValueError("Invalid primary metric")
        indexed[key] = row
    missing = expected - indexed.keys()
    if missing:
        return {
            "decision": "INCOMPLETE",
            "completed": len(indexed),
            "required": len(expected),
        }

    comparisons = []
    primary_pass = True
    ood_pass = True
    for domain in spec["domains"]:
        for block in spec["data_seeds"]:

            def mean(model, regime, domain=domain, block=block):
                return sum(
                    indexed[(domain, block, seed, model)]["regimes"][regime][
                        "position_rmse"
                    ]
                    for seed in spec["model_seeds"]
                ) / len(spec["model_seeds"])

            candidate = mean(CANDIDATE, "standard")
            for baseline in (*PRIMARY_BASELINES, GATE_ABLATION):
                reference = mean(baseline, "standard")
                threshold = (
                    spec["minimum_gate_ablation_improvement"]
                    if baseline == GATE_ABLATION
                    else spec["minimum_predecessor_improvement"]
                    if baseline == PREDECESSOR_BASELINE
                    else spec["minimum_primary_improvement"]
                )
                improvement = (reference - candidate) / max(reference, 1e-12)
                passed = improvement >= threshold
                primary_pass &= passed
                comparisons.append(
                    {
                        "kind": "primary",
                        "domain": domain,
                        "data_seed": block,
                        "baseline": baseline,
                        "candidate": candidate,
                        "reference": reference,
                        "relative_improvement": improvement,
                        "threshold": threshold,
                        "passes": passed,
                    }
                )
            for regime in spec["regimes"]:
                if regime == "standard":
                    continue
                shifted_candidate = mean(CANDIDATE, regime)
                reference = min(mean(name, regime) for name in PRIMARY_BASELINES)
                regression = (shifted_candidate - reference) / max(reference, 1e-12)
                passed = regression <= spec["maximum_ood_regression"]
                ood_pass &= passed
                comparisons.append(
                    {
                        "kind": "ood_noninferiority",
                        "domain": domain,
                        "data_seed": block,
                        "regime": regime,
                        "candidate": shifted_candidate,
                        "best_primary_baseline": reference,
                        "relative_regression": regression,
                        "threshold": spec["maximum_ood_regression"],
                        "passes": passed,
                    }
                )

    adequate = all(row["adequate"] for row in rows)
    candidate_gates = [
        (row["gate_mean"], row.get("diagnostics", {}).get("gate_std"))
        for row in rows
        if row["model"] == CANDIDATE
    ]
    gate_pass = bool(candidate_gates) and all(
        0.05 < mean < 0.95
        and isinstance(std, (int, float))
        and not isinstance(std, bool)
        and math.isfinite(std)
        and std >= spec["minimum_gate_std"]
        for mean, std in candidate_gates
    )
    decision = (
        "INCONCLUSIVE_UNDERTRAINED"
        if not adequate
        else "DEVELOPMENT_ADVANCE"
        if primary_pass and ood_pass and gate_pass
        else "NO_ADVANCE"
    )
    if spec["mode"] == "fixture":
        decision = "ENGINEERING_FIXTURE_ONLY"
    return {
        "decision": decision,
        "completed": len(indexed),
        "required": len(expected),
        "convergence_adequate": adequate,
        "primary_pass": primary_pass,
        "ood_pass": ood_pass,
        "gate_noncollapse_pass": gate_pass,
        "comparisons": comparisons,
        "claim_boundary": "Fresh synthetic development evidence; confirmation remains separately gated.",
    }


def paired_diagnostics(rows, spec, bootstrap_samples=2000):
    """Trajectory-paired descriptive intervals; never a confirmatory decision rule."""
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError("At least 100 deterministic bootstrap samples are required")
    indexed = {}
    for row in rows:
        key = tuple(
            row[name] for name in ("domain", "data_seed", "model_seed", "model")
        )
        if key in indexed:
            raise ValueError("Duplicate paired-diagnostic cell")
        indexed[key] = row

    def unit_values(domain, block, model, regime):
        by_seed = []
        for model_seed in spec["model_seeds"]:
            key = (domain, block, model_seed, model)
            if key not in indexed:
                raise ValueError(f"Missing paired-diagnostic cell: {key}")
            values = indexed[key].get("scene_position_mse", {}).get(regime)
            if not isinstance(values, list) or len(values) != spec["eval_scenes"]:
                raise ValueError(f"Invalid trajectory-unit vector: {key}/{regime}")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in values
            ):
                raise ValueError(f"Nonfinite trajectory-unit value: {key}/{regime}")
            by_seed.append(values)
        return [
            sum(seed_values[index] for seed_values in by_seed) / len(by_seed)
            for index in range(spec["eval_scenes"])
        ]

    comparisons = []
    baselines = (*PRIMARY_BASELINES, GATE_ABLATION)
    for domain in spec["domains"]:
        for block in spec["data_seeds"]:
            for regime in spec["regimes"]:
                candidate = unit_values(domain, block, CANDIDATE, regime)
                for baseline in baselines:
                    reference = unit_values(domain, block, baseline, regime)

                    def relative(
                        indices, candidate=candidate, reference=reference
                    ):
                        candidate_rmse = math.sqrt(
                            sum(candidate[index] for index in indices) / len(indices)
                        )
                        reference_rmse = math.sqrt(
                            sum(reference[index] for index in indices) / len(indices)
                        )
                        return (
                            reference_rmse - candidate_rmse
                        ) / max(reference_rmse, 1e-12)

                    count = len(candidate)
                    observed = relative(range(count))
                    seed_material = (
                        f"{spec['study']}|{domain}|{block}|{regime}|{baseline}|"
                        f"{bootstrap_samples}"
                    ).encode()
                    seed = int.from_bytes(
                        hashlib.sha256(seed_material).digest()[:8], "little"
                    )
                    generator = random.Random(seed)
                    bootstrap = sorted(
                        relative([generator.randrange(count) for _ in range(count)])
                        for _ in range(bootstrap_samples)
                    )
                    lower = bootstrap[int(0.025 * (bootstrap_samples - 1))]
                    upper = bootstrap[int(0.975 * (bootstrap_samples - 1))]
                    comparisons.append(
                        {
                            "domain": domain,
                            "data_seed": block,
                            "regime": regime,
                            "baseline": baseline,
                            "trajectories": count,
                            "model_seeds_nested": len(spec["model_seeds"]),
                            "relative_rmse_improvement": observed,
                            "paired_trajectory_bootstrap_95": [lower, upper],
                        }
                    )

    aggregates = []
    for domain in spec["domains"]:
        for regime in spec["regimes"]:
            for baseline in baselines:
                values = [
                    row["relative_rmse_improvement"]
                    for row in comparisons
                    if row["domain"] == domain
                    and row["regime"] == regime
                    and row["baseline"] == baseline
                ]
                aggregates.append(
                    {
                        "domain": domain,
                        "regime": regime,
                        "baseline": baseline,
                        "independent_blocks": len(values),
                        "block_mean": sum(values) / len(values),
                        "block_min": min(values),
                        "block_max": max(values),
                        "block_values": values,
                    }
                )
    return {
        "format": "ripii-v3-paired-diagnostics-v1",
        "bootstrap_samples": bootstrap_samples,
        "comparisons": comparisons,
        "independent_block_summaries": aggregates,
        "decision_use": "DESCRIPTIVE_ONLY_DOES_NOT_CHANGE_PREDECLARED_GATE",
        "claim_boundary": (
            "Trajectory intervals are conditional within each generated block; model "
            "seeds are nested and four development blocks do not establish "
            "confirmatory population inference."
        ),
    }
