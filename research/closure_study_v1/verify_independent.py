"""Independently recompute a completed closure phase from retained arrays.

Run with ``PHASE_DIRECTORY --output REPORT.json``. The report must be outside
the immutable phase directory. ``--self-test`` uses hand-constructed arrays and
no study outcomes. This script imports no RIPII module, prediction routine, or
analysis helper. Torch is used only to decode retained tensor files; metrics
are independently accumulated per scene in NumPy float64.

This verifies numerical reporting and identities, not a new scientific decision
rule. In particular, it does not retrain, replay a model, or reclassify H2.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

SPLITS = ("test", "more_objects", "composition", "fast")
ALL_DATA_SPLITS = ("train", "validation", *SPLITS)
PREDICTION_SPLITS = (*SPLITS, "noisy_test")
ATOL = 2e-7
RTOL = 5e-6


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def array(value: Any, location: str, *, floating: bool = True) -> np.ndarray:
    require(isinstance(value, torch.Tensor), f"{location}: expected tensor")
    require(value.layout == torch.strided and value.device.type == "cpu",
            f"{location}: expected dense CPU tensor")
    if floating:
        require(value.dtype in (torch.float32, torch.float64),
                f"{location}: expected float32/float64")
    result = value.detach().numpy()
    require(bool(np.isfinite(result).all()), f"{location}: nonfinite values")
    return result.astype(np.float64) if floating else result


def errors_from_arrays(prediction: np.ndarray, target: np.ndarray,
                       mask: np.ndarray) -> tuple[dict, dict]:
    """Accumulate live coordinates scene by scene, excluding observation zero."""
    require(target.ndim == 4 and target.shape[-1] == 6, "invalid target shape")
    require(prediction.shape == target.shape, "prediction/target shape mismatch")
    batch, observations, objects, _ = target.shape
    require(batch > 0 and observations > 1, "empty evaluation batch or horizon")
    require(mask.shape == (batch, objects) and mask.dtype == np.bool_, "invalid mask")
    require(bool(np.isfinite(prediction).all() and np.isfinite(target).all()),
            "nonfinite prediction or target")
    counts = mask.sum(axis=1, dtype=np.int64)
    require(bool((counts > 0).all()), "empty live scene")
    horizon = observations - 1
    position_sse = np.zeros((batch, horizon), dtype=np.float64)
    velocity_sse = np.zeros_like(position_sse)
    denominators = []
    for index in range(batch):
        # Selecting live indices, rather than multiplying by a broadcast mask,
        # independently checks that padded coordinates never enter a denominator.
        live = np.flatnonzero(mask[index])
        difference = (prediction[index, 1:][:, live, :4].astype(np.float64)
                      - target[index, 1:][:, live, :4].astype(np.float64))
        position_sse[index] = np.sum(difference[:, :, :2] ** 2, axis=(1, 2), dtype=np.float64)
        velocity_sse[index] = np.sum(difference[:, :, 2:] ** 2, axis=(1, 2), dtype=np.float64)
        denominators.append(2 * horizon * int(counts[index]))
    denominator = sum(denominators)
    pos_total = math.fsum(position_sse.reshape(-1).tolist())
    vel_total = math.fsum(velocity_sse.reshape(-1).tolist())
    require(math.isfinite(pos_total) and math.isfinite(vel_total), "squared error overflow")
    metrics = {
        "position_rmse": math.sqrt(pos_total / denominator),
        "velocity_rmse": math.sqrt(vel_total / denominator),
    }
    for endpoint in (1, 4, 16, 32):
        if endpoint <= horizon:
            total = math.fsum(position_sse[:, endpoint - 1].tolist())
            metrics[f"position_rmse_h{endpoint}"] = math.sqrt(total / (2 * int(counts.sum())))
    for count in sorted(set(counts.tolist())):
        selected = np.flatnonzero(counts == count)
        total = math.fsum(position_sse[selected].reshape(-1).tolist())
        metrics[f"position_rmse_n{count}"] = math.sqrt(total / (2 * horizon * count * len(selected)))
    per_scene = np.sqrt(position_sse.sum(axis=1) / np.asarray(denominators))
    metrics["worst_scene_position_rmse"] = float(per_scene.max())
    raw = {
        "position_sse_per_scene_time": position_sse,
        "velocity_sse_per_scene_time": velocity_sse,
        "per_scene_position_rmse": per_scene,
        "counts": counts,
        "denominator_live_coordinates_time": denominator,
        "horizon": horizon,
    }
    return metrics, raw


def summary(values: list[float]) -> dict:
    require(bool(values) and all(math.isfinite(value) for value in values),
            "cannot aggregate empty or nonfinite observations")
    return {"n": len(values), "mean": math.fsum(values) / len(values),
            "min": min(values), "max": max(values)}


def paired_statistics(pairs: list[tuple[int, int]], values: dict,
                      candidate: str, control: str, statistics: dict,
                      *, iid: bool = False, confirmation: bool = False) -> dict:
    require(bool(pairs) and len(pairs) == len(set(pairs)) and set(values) == set(pairs),
            "duplicate, missing, or unmatched independent pairs")
    rows, logarithms = [], []
    for pair in pairs:
        require(candidate in values[pair] and control in values[pair],
                f"{pair}: missing candidate or paired control")
        key = "iid_position_rmse" if iid else "ood_position_rmse"
        left, right = (values[pair][arm][key] for arm in (candidate, control))
        require(all(math.isfinite(x) and x >= 0 for x in (left, right)), "invalid paired RMSE")
        ratio = left / right if left > 0 and right > 0 else None
        valid = ratio is not None and ratio > 0 and math.isfinite(ratio)
        logarithm = math.log(left) - math.log(right) if valid else None
        rows.append({"data_seed": pair[0], "model_seed": pair[1],
                     "candidate_rmse": left, "control_rmse": right,
                     "ratio": ratio if valid else None, "log_ratio": logarithm})
        if logarithm is not None:
            logarithms.append(logarithm)
    result = {"candidate": candidate, "control": control, "unit_effects": rows,
              "n_pairs": len(logarithms), "geometric_ratio": None,
              "interval_95": None, "primary_sign_flip_p": None}
    if len(logarithms) != len(pairs):
        return result  # No pseudocount and no selected positive-error subset.
    logs = np.asarray(logarithms, dtype=np.float64)
    result["geometric_ratio"] = math.exp(math.fsum(logarithms) / len(logarithms))
    rng = np.random.default_rng(statistics["bootstrap_seed"])
    indices = rng.integers(len(pairs), size=(statistics["bootstrap_repetitions"], len(pairs)))
    bootstrap_ratios = np.exp(logs[indices].sum(axis=1) / len(pairs))
    result["interval_95"] = np.percentile(bootstrap_ratios, statistics["interval_percentiles"]).tolist()
    if confirmation and not iid:
        centered = [value - math.log(statistics["practical_ratio"]) for value in logarithms]
        observed = abs(math.fsum(centered) / len(centered))
        threshold = observed - 1e-12 * max(1.0, observed)
        extreme = 0
        for bits in range(1 << len(centered)):
            signed = [value if bits & (1 << index) else -value
                      for index, value in enumerate(centered)]
            extreme += abs(math.fsum(signed) / len(signed)) >= threshold
        result["primary_sign_flip_p"] = extreme / (1 << len(centered))
    return result


class IndependentAudit:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.inputs: dict[str, dict] = {}
        self.errors: list[dict] = []
        self.comparisons = 0
        self.max_absolute_error = 0.0
        self.max_scaled_error = 0.0
        self.worst_location: str | None = None
        self.manifest: dict | None = None
        self.result: dict = {"format": "ripii-closure-independent-verification-v1",
                             "phase_directory": str(self.directory)}

    def read(self, relative: str, *, tensor: bool = False) -> Any:
        part = Path(relative)
        require(not part.is_absolute() and ".." not in part.parts, "unsafe evidence path")
        path = self.directory / part
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular input: {relative}")
        require(self.directory in path.resolve().parents, f"input escapes phase: {relative}")
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        self.inputs[relative] = {"sha256": digest, "bytes": len(encoded)}
        if self.manifest is not None and relative != "manifest.json":
            require(self.manifest.get(relative) == self.inputs[relative],
                    f"input differs from phase manifest: {relative}")
        if tensor:
            return torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
        return json.loads(encoded)

    def equal(self, location: str, actual: Any, expected: Any) -> None:
        self.comparisons += 1
        if actual != expected:
            self.errors.append({"location": location, "kind": "identity",
                                "reported": actual, "recomputed": expected})

    def numeric(self, location: str, actual: Any, expected: Any) -> None:
        self.comparisons += 1
        if expected is None:
            if actual is not None:
                self.errors.append({"location": location, "kind": "unavailable",
                                    "reported": actual, "recomputed": None})
            return
        try:
            observed = np.asarray(actual, dtype=np.float64)
            reference = np.asarray(expected, dtype=np.float64)
        except (TypeError, ValueError):
            self.errors.append({"location": location, "kind": "nonnumeric"})
            return
        if (observed.shape != reference.shape or not np.isfinite(observed).all()
                or not np.isfinite(reference).all()):
            self.errors.append({"location": location, "kind": "shape_or_finite"})
            return
        difference = np.abs(observed - reference)
        allowed = ATOL + RTOL * np.abs(reference)
        absolute = float(difference.max(initial=0))
        scaled = float((difference / allowed).max(initial=0))
        self.max_absolute_error = max(self.max_absolute_error, absolute)
        if scaled > self.max_scaled_error:
            self.max_scaled_error, self.worst_location = scaled, location
        if scaled > 1:
            self.errors.append({"location": location, "kind": "numerical",
                                "max_absolute_error": absolute,
                                "max_tolerance_fraction": scaled})

    def metrics(self, location: str, actual: dict, expected: dict) -> None:
        require(isinstance(actual, dict), f"{location}: missing metric mapping")
        for metric, value in expected.items():
            require(metric in actual, f"{location}: missing metric {metric}")
            self.numeric(f"{location}/{metric}", actual[metric], value)

    def dataset(self, prefix: str, split: str, config: dict, data_seed: int,
                seen_initial: set[str], provenance: dict) -> dict:
        data = self.read(f"{prefix}/data/{split}.pt", tensor=True)
        require(isinstance(data, dict) and set(data) == {"states", "actions", "mask", "ids"},
                f"{prefix}/{split}: invalid dataset keys")
        target = array(data["states"], f"{prefix}/{split}/states")
        actions = array(data["actions"], f"{prefix}/{split}/actions")
        mask = array(data["mask"], f"{prefix}/{split}/mask", floating=False)
        ids = array(data["ids"], f"{prefix}/{split}/ids", floating=False)
        scenes = config["train_scenes"] if split == "train" else (
            config["validation_scenes"] if split == "validation" else config["evaluation_scenes"])
        horizon = config["train_horizon"] if split == "train" else config["evaluation_horizon"]
        objects = config["max_objects"]
        require(target.shape == (scenes, horizon + 1, objects, 6), f"{prefix}/{split}: target shape")
        require(actions.shape == (scenes, horizon, objects, 2), f"{prefix}/{split}: action shape")
        require(mask.shape == (scenes, objects) and mask.dtype == np.bool_, f"{prefix}/{split}: mask")
        require(ids.shape == (scenes,) and ids.dtype == np.int64, f"{prefix}/{split}: IDs")
        require(bool(mask.any(axis=1).all()), f"{prefix}/{split}: empty scene")
        require(np.count_nonzero(target.transpose(0, 2, 1, 3)[~mask]) == 0
                and np.count_nonzero(actions.transpose(0, 2, 1, 3)[~mask]) == 0,
                f"{prefix}/{split}: nonzero data padding")
        properties = target[..., 4:6].transpose(0, 2, 1, 3)[mask]
        require(bool((properties > 0).all())
                and np.array_equal(properties, np.broadcast_to(properties[:, :1], properties.shape)),
                f"{prefix}/{split}: invalid or changing live properties")
        # This identity formula belongs to the disclosed synthetic generator;
        # it catches files swapped between regimes even if array shapes match.
        offset = ALL_DATA_SPLITS.index(split) * 1_000_003
        expected_ids = np.arange(scenes, dtype=np.int64) + (data_seed + offset) * 1_000_000
        self.equal(f"{prefix}/{split}/split_qualified_ids", ids.tolist(), expected_ids.tolist())
        spec = provenance["splits"][split]["spec"]
        for key, value in {"split": split, "seed": data_seed, "scenes": scenes,
                           "horizon": horizon, "max_objects": objects,
                           "physics": config["physics"]}.items():
            self.equal(f"{prefix}/{split}/spec/{key}", spec.get(key), value)
        for index in range(scenes):
            digest = hashlib.sha256(target[index, 0].tobytes() + mask[index].tobytes()).hexdigest()
            require(digest not in seen_initial, f"{prefix}/{split}: duplicate initial scene across/within splits")
            seen_initial.add(digest)
        return {"target": target, "mask": mask, "ids": ids}

    def checkpoint(self, prefix: str, arm: dict, model_seed: int,
                   protocol: dict, source_hash: str, record: dict) -> dict:
        if arm["kind"] == "oracle":
            self.equal(f"{prefix}/parameters", record.get("parameters"), 0)
            return {"parameters": 0, "kind": "oracle"}
        payload = self.read(f"{prefix}/selected.pt", tensor=True)
        require(isinstance(payload, dict) and payload.get("format") == "ripii-closure-v1",
                f"{prefix}: invalid selected checkpoint")
        physical = protocol["dataset"]["physics"]
        expected_config = {**protocol["model"],
                           **{key: physical[key] for key in ("dt", "drag", "stiffness", "damping")},
                           **{key: arm[key] for key in ("kind", "closure", "substeps")}}
        self.equal(f"{prefix}/config", payload.get("config"), expected_config)
        for key, value in {"seed": model_seed, "protocol_sha256": canonical_sha256(protocol),
                           "source_sha256": source_hash,
                           "completed_updates": record["selected_update"]}.items():
            self.equal(f"{prefix}/checkpoint/{key}", payload.get(key), value)
        weights = payload["model_state"]
        require(isinstance(weights, dict), f"{prefix}: invalid state dictionary")
        signature = array(weights["_config_signature"], f"{prefix}/signature", floating=False)
        expected_signature = list(bytes.fromhex(canonical_sha256(expected_config)))
        self.equal(f"{prefix}/configuration_signature", signature.tolist(), expected_signature)
        require(signature.dtype == np.uint8, f"{prefix}: signature dtype")
        parameters = 0
        for name, value in weights.items():
            if name != "_config_signature":
                values = array(value, f"{prefix}/{name}")
                parameters += values.size
        hidden = expected_config["hidden"]
        encoder = 2 * hidden**2 + 9 * hidden
        head = 3 * hidden**2 + 6 * hidden + 1
        expected_parameters = {"local": encoder + head, "global": encoder + 2 * head,
                               "hierarchy": encoder + 2 * head, "mean": encoder + head + 1,
                               "law": 1, "prior": 0}[arm["kind"]]
        self.equal(f"{prefix}/parameter_count_from_weights", record.get("parameters"), parameters)
        self.equal(f"{prefix}/parameter_count_from_spec", parameters, expected_parameters)
        result = {"parameters": parameters, "config": payload["config"],
                  "selected_update": payload["completed_updates"]}
        if arm["kind"] in ("mean", "law"):
            coefficient = float(np.logaddexp(0.0, array(weights["rho"], f"{prefix}/rho")))
            self.numeric(f"{prefix}/fitted_coupling", record.get("fitted_coupling"), coefficient)
            result["fitted_coupling"] = coefficient
        return result

    def verify(self) -> None:
        # Refuse an unfinished phase before opening predictions or phase outcomes.
        status = self.read("status.json")
        require(status.get("state") == "complete", "phase status is not complete")
        manifest = self.read("manifest.json")
        require(manifest.get("format") == "ripii-closure-evidence-manifest-v1",
                "invalid evidence manifest")
        self.manifest = manifest["files"]
        require(isinstance(self.manifest, dict) and bool(self.manifest), "empty manifest")
        require(self.manifest.get("status.json") == self.inputs["status.json"], "status hash mismatch")
        payload = self.read("phase_results.json")
        protocol = self.read("protocol.json")
        analysis = self.read("analysis/analysis.json")
        source = self.read("source.json")
        phase = payload.get("phase")
        require(payload.get("format") == "ripii-closure-phase-results-v1"
                and payload.get("complete") is True, "phase results are not complete")
        require(phase in ("dry_run", "development", "confirmation"), "unknown phase")
        require(protocol.get("format") == "ripii-closure-study-protocol-v1", "unknown protocol")
        require(protocol["dataset"]["evaluation_splits"] == list(SPLITS), "regime set/order mismatch")
        self.result.update(phase=phase, protocol_sha256=canonical_sha256(protocol),
                           reported_source_sha256=source["source_sha256"])
        self.equal("phase/protocol", payload.get("protocol"), protocol)
        self.equal("status/protocol", status.get("protocol_sha256"), canonical_sha256(protocol))
        self.equal("status/events", [event.get("state") for event in status.get("events", [])],
                   ["planned", "running", "complete"])
        self.equal("source/record_digest", source.get("source_sha256"), canonical_sha256(source["files"]))
        for relative, digest in source["files"].items():
            path = self.directory / "source" / relative
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                    "unsafe source snapshot path")
            require(path.is_file() and not path.is_symlink(), f"missing source snapshot {relative}")
            require(self.directory in path.resolve().parents, f"source snapshot escapes phase: {relative}")
            encoded = path.read_bytes()
            key = f"source/{relative}"
            self.inputs[key] = {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
            self.equal(f"source/{relative}", self.inputs[key]["sha256"], digest)
            self.equal(f"manifest/{key}", self.manifest.get(key), self.inputs[key])
        verifier_key = "research/closure_study_v1/verify_independent.py"
        retained_verifier = source["files"].get(verifier_key)
        self.result["verifier_in_retained_source"] = retained_verifier is not None
        if retained_verifier is not None:
            self.equal("source/active_independent_verifier", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       retained_verifier)
        seeds = [protocol["seeds"][phase]] if phase == "dry_run" else protocol["seeds"][phase]
        pairs = [(item["data"], item["model"]) for item in seeds]
        require(len(pairs) == {"dry_run": 1, "development": 3, "confirmation": 6}[phase]
                and len(set(pairs)) == len(pairs) and len({p[0] for p in pairs}) == len(pairs),
                "invalid independent seed grid")
        units = payload["units"]
        actual_pairs = [(unit["data_seed"], unit["model_seed"]) for unit in units]
        require(len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == set(pairs),
                "duplicate, missing, unexpected, or unmatched phase units")
        by_pair = dict(zip(actual_pairs, units, strict=True))
        arm_specs = {arm["id"]: arm for arm in protocol["arms"]}
        require(len(arm_specs) == len(protocol["arms"]), "duplicate protocol arm IDs")
        control = protocol["compute_control"]
        require(control["id"] not in arm_specs, "duplicate compute control")
        arm_specs[control["id"]] = {**arm_specs[control["source_arm"]], "id": control["id"]}
        expected_arms = set(arm_specs)
        self.equal("analysis/format", analysis.get("format"), "ripii-closure-analysis-v1")
        self.equal("analysis/phase", analysis.get("phase"), phase)
        self.equal("analysis/complete", analysis.get("complete"), True)
        self.equal("analysis/expected_units", analysis.get("expected_units"), len(pairs))
        self.equal("analysis/observed_units", analysis.get("observed_units"), len(pairs))
        self.equal("analysis/split_order", analysis.get("split_order"), list(SPLITS))
        self.equal("analysis/arm_order", analysis.get("arm_order"), list(arm_specs))
        self.equal("analysis/statistics", analysis.get("statistics"), protocol["statistics"])
        analysis_units = analysis["unit_outcomes"]
        published_pairs = [(u["data_seed"], u["model_seed"]) for u in analysis_units]
        require(len(published_pairs) == len(set(published_pairs)) and set(published_pairs) == set(pairs),
                "analysis contains unmatched or duplicate unit pairs")
        published_by_pair = dict(zip(published_pairs, analysis_units, strict=True))
        values: dict = {}
        recomputed_units = []
        checked_splits = 0
        for pair in pairs:
            unit, published = by_pair[pair], published_by_pair[pair]
            prefix = f"units/{pair[0]}"
            require(set(unit["arms"]) == expected_arms and set(published["arms"]) == expected_arms,
                    f"{prefix}: missing or unexpected arm")
            self.equal(f"{prefix}/unit_record", self.read(f"{prefix}/unit.json"), unit)
            provenance = self.read(f"{prefix}/data/data_manifest.json")
            self.equal(f"{prefix}/data_seed", provenance.get("data_seed"), pair[0])
            require(set(provenance["splits"]) == set(ALL_DATA_SPLITS), f"{prefix}: dataset split grid")
            seen_initial: set[str] = set()
            data = {split: self.dataset(prefix, split, protocol["dataset"], pair[0], seen_initial, provenance)
                    for split in ALL_DATA_SPLITS}
            values[pair] = {}
            current = {"data_seed": pair[0], "model_seed": pair[1], "arms": {}}
            for name, arm in arm_specs.items():
                location = f"{prefix}/{name}"
                record = unit["arms"][name]
                arm_status = self.read(f"{location}/status.json")
                self.equal(f"{location}/status", arm_status.get("state"), "complete")
                self.equal(f"{location}/status_protocol", arm_status.get("protocol_sha256"), canonical_sha256(protocol))
                self.equal(f"{location}/result", self.read(f"{location}/result.json"), record)
                checkpoint_record = self.checkpoint(location, arm, pair[1], protocol, source["source_sha256"], record)
                evaluation = self.read(f"{location}/evaluation.json")
                predictions = self.read(f"{location}/predictions.pt", tensor=True)
                raw = self.read(f"{location}/raw_metrics.pt", tensor=True)
                require(set(predictions) == set(PREDICTION_SPLITS) and set(raw) == set(PREDICTION_SPLITS),
                        f"{location}: missing/unexpected prediction or raw regime")
                require(set(record["metrics"]) == set(SPLITS)
                        and set(evaluation["metrics"]) == set(SPLITS), f"{location}: reported split mismatch")
                recomputed = {}
                for split in PREDICTION_SPLITS:
                    dataset = data["test" if split == "noisy_test" else split]
                    prediction = array(predictions[split], f"{location}/{split}/prediction")
                    target, mask = dataset["target"], dataset["mask"]
                    metrics, raw_expected = errors_from_arrays(prediction, target, mask)
                    require(np.count_nonzero(prediction.transpose(0, 2, 1, 3)[~mask]) == 0,
                            f"{location}/{split}: nonzero prediction padding")
                    require(np.array_equal(prediction[..., 4:6], target[..., 4:6]),
                            f"{location}/{split}: changed radius or mass")
                    if split != "noisy_test":
                        require(np.array_equal(prediction[:, 0], target[:, 0]),
                                f"{location}/{split}: initial observation mismatch")
                    for label, container in (("phase", record), ("evaluation", evaluation),
                                             ("analysis", published["arms"][name])):
                        retained = container["noise_metrics"] if split == "noisy_test" else container["metrics"][split]
                        self.metrics(f"{location}/{split}/{label}", retained, metrics)
                    for key in ("position_sse_per_scene_time", "velocity_sse_per_scene_time", "per_scene_position_rmse"):
                        self.numeric(f"{location}/{split}/raw/{key}", array(raw[split][key], key), raw_expected[key])
                    self.equal(f"{location}/{split}/raw/counts", array(raw[split]["counts"], "counts", floating=False).tolist(),
                               raw_expected["counts"].tolist())
                    self.equal(f"{location}/{split}/raw/ids", array(raw[split]["ids"], "ids", floating=False).tolist(),
                               dataset["ids"].tolist())
                    recomputed[split] = {**metrics, "denominator_live_coordinates_time": raw_expected["denominator_live_coordinates_time"],
                                         "counts": raw_expected["counts"].tolist(), "horizon": raw_expected["horizon"]}
                    checked_splits += 1
                ood = math.fsum(recomputed[split]["position_rmse"] for split in SPLITS[1:]) / 3
                self.numeric(f"{location}/analysis/ood_position_rmse", published["arms"][name]["ood_position_rmse"], ood)
                values[pair][name] = {"ood_position_rmse": ood, "iid_position_rmse": recomputed["test"]["position_rmse"],
                                      "metrics": recomputed}
                current["arms"][name] = {"checkpoint": checkpoint_record, "metrics": recomputed,
                                         "ood_position_rmse": ood}
            hierarchy_time = unit["arms"][control["target_arm"]]["optimization_seconds"]
            flat_time = unit["arms"][control["id"]]["optimization_seconds"]
            require(hierarchy_time > 0 and flat_time >= 0, f"{prefix}: invalid optimization times")
            self.numeric(f"{prefix}/compute_coverage", unit["arms"][control["id"]]["compute_coverage_ratio"],
                         flat_time / hierarchy_time)
            recomputed_units.append(current)
        for name in arm_specs:
            arm_summary = analysis["arm_summaries"][name]
            for metric, value in summary([values[pair][name]["ood_position_rmse"] for pair in pairs]).items():
                self.numeric(f"analysis/{name}/ood/{metric}", arm_summary["ood_position_rmse"][metric], value)
            for split in PREDICTION_SPLITS:
                metric_names = sorted({metric for pair in pairs for metric in values[pair][name]["metrics"][split]
                                       if metric not in ("counts", "horizon", "denominator_live_coordinates_time")})
                for metric in metric_names:
                    observations = [values[pair][name]["metrics"][split][metric] for pair in pairs
                                    if metric in values[pair][name]["metrics"][split]]
                    published_summary = (arm_summary["noise_metrics"] if split == "noisy_test"
                                         else arm_summary["regimes"][split])[metric]
                    for key, value in summary(observations).items():
                        self.numeric(f"analysis/{name}/{split}/{metric}/{key}", published_summary[key], value)
        statistics = protocol["statistics"]
        candidate = statistics["primary_candidate"]
        controls = [statistics["primary_control"], *statistics["required_other_controls"]]
        require(set(analysis["comparisons"]) == set(controls), "analysis comparison/control mismatch")
        comparisons = {}
        for name in [*controls, "iid"]:
            iid = name == "iid"
            control_name = statistics["primary_control"] if iid else name
            recomputed = paired_statistics(pairs, values, candidate, control_name, statistics,
                                           iid=iid, confirmation=phase == "confirmation" and name == controls[0])
            published_comparison = analysis["iid_comparison"] if iid else analysis["comparisons"][name]
            for key in ("candidate", "control", "n_pairs"):
                self.equal(f"comparison/{name}/{key}", published_comparison.get(key), recomputed[key])
            rows = published_comparison["unit_effects"]
            row_pairs = [(row["data_seed"], row["model_seed"]) for row in rows]
            require(len(row_pairs) == len(set(row_pairs)) and set(row_pairs) == set(pairs),
                    f"comparison/{name}: unmatched effect pairs")
            rows_by_pair = dict(zip(row_pairs, rows, strict=True))
            for row in recomputed["unit_effects"]:
                pair = (row["data_seed"], row["model_seed"])
                for key in ("candidate_rmse", "control_rmse", "ratio", "log_ratio"):
                    self.numeric(f"comparison/{name}/{pair}/{key}", rows_by_pair[pair].get(key), row[key])
            for key in ("geometric_ratio", "interval_95", "primary_sign_flip_p"):
                self.numeric(f"comparison/{name}/{key}", published_comparison.get(key), recomputed[key])
            comparisons[name] = recomputed
        self.result.update(independent_units=len(pairs), arms_per_unit=len(arm_specs),
                           prediction_splits_checked=checked_splits,
                           recomputed_units=recomputed_units, comparisons=comparisons,
                           published_h2_not_reclassified=analysis.get("decision_h2"))

    def report(self) -> dict:
        return {**self.result, "passed": not self.errors, "errors": self.errors,
                "scalar_or_array_comparisons": self.comparisons,
                "max_absolute_discrepancy": self.max_absolute_error,
                "max_tolerance_fraction": self.max_scaled_error,
                "worst_scaled_discrepancy_location": self.worst_location,
                "tolerance": {"absolute": ATOL, "relative": RTOL,
                              "reason": "independent float64 accumulation compared with retained float32 arithmetic"},
                "input_files": self.inputs,
                "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Saved-array RMSEs, endpoint/count strata, raw per-scene errors, equally weighted OOD regime means, paired ratios, prescribed bootstrap/sign flips, parameter counts, checkpoint identities, and input/source hashes; no retraining, model replay, new decision rule, or authenticated provenance."}


def self_test() -> dict:
    target = np.zeros((2, 3, 3, 6), dtype=np.float64)
    prediction = target.copy()
    mask = np.asarray([[True, False, False], [True, True, False]], dtype=np.bool_)
    prediction[0, 1:, 0, :2] = 1
    prediction[1, 1:, :2, :2] = 3
    prediction[0, 1:, 0, 2:4] = 2
    prediction[1, 1:, :2, 2:4] = 4
    prediction[:, 0, :, :4] = 1e20  # Observation zero must be excluded.
    prediction.transpose(0, 2, 1, 3)[~mask] = 1e30  # Padding must be excluded.
    metrics, raw = errors_from_arrays(prediction, target, mask)
    require(raw["denominator_live_coordinates_time"] == 12, "hand-case denominator")
    np.testing.assert_allclose(metrics["position_rmse"], math.sqrt(19 / 3), rtol=1e-14)
    np.testing.assert_allclose(metrics["velocity_rmse"], math.sqrt(12), rtol=1e-14)
    np.testing.assert_allclose(raw["per_scene_position_rmse"], [1, 3], rtol=1e-14)
    np.testing.assert_allclose(metrics["position_rmse_n1"], 1)
    np.testing.assert_allclose(metrics["position_rmse_n2"], 3)
    # Unequal scene counts distinguish pooled-coordinate RMSE from the mean of scene RMSEs.
    require(not math.isclose(metrics["position_rmse"], 2), "scene-mean denominator test")
    pairs = [(11, 21), (12, 22)]
    values = {pairs[0]: {"candidate": {"ood_position_rmse": (1 + 2 + 6) / 3},
                         "control": {"ood_position_rmse": (2 + 4 + 12) / 3}},
              pairs[1]: {"candidate": {"ood_position_rmse": (2 + 4 + 8) / 3},
                         "control": {"ood_position_rmse": (1 + 2 + 4) / 3}}}
    statistics = {"bootstrap_seed": 46001, "bootstrap_repetitions": 10000,
                  "interval_percentiles": [2.5, 97.5], "practical_ratio": 0.95}
    paired = paired_statistics(pairs, values, "candidate", "control", statistics)
    np.testing.assert_allclose(paired["geometric_ratio"], 1, atol=1e-15)
    np.testing.assert_allclose([row["ratio"] for row in paired["unit_effects"]], [0.5, 2])
    caught = 0
    for bad_pairs, bad_values in (([pairs[0], pairs[0]], values), (pairs, {pairs[0]: values[pairs[0]]})):
        try:
            paired_statistics(bad_pairs, bad_values, "candidate", "control", statistics)
        except ValueError:
            caught += 1
    require(caught == 2, "unmatched/duplicate-pair rejection")
    values[pairs[0]]["candidate"]["ood_position_rmse"] = 0
    unavailable = paired_statistics(pairs, values, "candidate", "control", statistics)
    require(unavailable["geometric_ratio"] is None and unavailable["n_pairs"] == 1,
            "zero error must not cause a selected-subset aggregate")
    audit = IndependentAudit(Path.cwd())
    audit.numeric("wrong_denominator", 2, metrics["position_rmse"])
    require(bool(audit.errors), "numeric mismatch must be detected")
    return {"passed": True, "checks": ["unequal-count coordinate denominator", "padding excluded",
                                        "initial observation excluded", "velocity RMSE", "per-scene/count metrics",
                                        "OOD arithmetic means", "geometric paired mean", "duplicate/missing pairs",
                                        "zero-error unavailability", "numerical mismatch detection"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_directory", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        require(args.phase_directory is None and args.output is None,
                "self-test reads no phase and writes no artifact")
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
        return 0
    if args.phase_directory is None or args.output is None:
        parser.error("a completed phase directory and --output are required")
    phase = args.phase_directory.resolve()
    require(not args.output.is_symlink(), "report path must not be a symlink")
    output = args.output.resolve()
    require(phase.is_dir() and not args.phase_directory.is_symlink(), "phase must be a regular directory")
    require(output != phase and phase not in output.parents, "report must be outside the retained phase directory")
    require(not output.exists(), "refusing to overwrite an existing independent report")
    audit = IndependentAudit(phase)
    try:
        audit.verify()
    except Exception as exc:
        audit.errors.append({"location": "verification", "kind": type(exc).__name__, "message": str(exc)})
    report = audit.report()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(output),
                      "errors": len(report["errors"]),
                      "max_absolute_discrepancy": report["max_absolute_discrepancy"]}, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
