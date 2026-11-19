"""Pure-standard-library protocol, training adequacy and paired inference."""

import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    def bad(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), parse_constant=bad)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def adequate(curve, selected_train, cv_train, protocol):
    if not curve or any(
        not math.isfinite(x) or x < 0 for x in (selected_train, cv_train)
    ):
        return {"pass": False, "reason": "INVALID_OR_EMPTY"}
    full = [r for r in curve if r["horizon"] == protocol["forecast"]]
    window = protocol["plateau_window"]
    if curve[-1]["step"] < protocol["minimum_steps"] or len(full) < window + 1:
        return {"pass": False, "reason": "INSUFFICIENT_FULL_HORIZON_TRAINING"}
    gains = {}
    for key in ("train_rmse", "validation_rmse"):
        values = [r[key] for r in full]
        if any(not math.isfinite(x) or x < 0 for x in values):
            return {"pass": False, "reason": "NONFINITE_CURVE"}
        earlier = min(values[:-window])
        gains[key] = (earlier - min(values)) / max(earlier, 1e-12)
    ratio = selected_train / max(cv_train, 1e-12)
    passed = ratio <= protocol["maximum_training_cv_ratio"] and all(
        value <= protocol["maximum_tail_improvement"] for value in gains.values()
    )
    return {
        "pass": passed,
        "reason": "ADEQUATE" if passed else "UNDERTRAINED",
        "tail_gains": gains,
        "training_cv_ratio": ratio,
    }


def sign_permutation(log_ratios):
    if not log_ratios or any(not math.isfinite(x) for x in log_ratios):
        raise ValueError("Finite paired independent-block values required")
    observed = sum(log_ratios)
    counts = sum(
        sum(s * x for s, x in zip(signs, log_ratios, strict=True)) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(log_ratios))
    )
    return counts / (2 ** len(log_ratios))


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def summarize(rows, protocol):
    expected = {
        (d, b, s, m)
        for d in protocol["domains"]
        for b in protocol["data_blocks"]
        for s in protocol["model_seeds"]
        for m in protocol["models"]
    }
    index = {}
    for row in rows:
        key = tuple(row[k] for k in ("domain", "block", "seed", "model"))
        if key not in expected or key in index:
            raise ValueError("Unexpected or duplicate evaluation")
        if not row["adequacy"]["pass"]:
            raise ValueError("Protected evaluation contains inadequate fit")
        for regime in protocol["regimes"]:
            value = row["metrics"][regime]["position_rmse"]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid metric")
        index[key] = row
    if set(index) != expected:
        return {
            "decision": "INCOMPLETE",
            "completed": len(index),
            "required": len(expected),
        }
    comparisons = []
    for domain in protocol["domains"]:
        for baseline in [m for m in protocol["models"] if m != "radial_mixture"] + [
            "train_mean_gate"
        ]:
            for regime in protocol["regimes"]:
                gains, ratios = [], []
                for block in protocol["data_blocks"]:
                    candidate, reference = [], []
                    for seed in protocol["model_seeds"]:
                        row = index[(domain, block, seed, "radial_mixture")]
                        candidate.append(row["metrics"][regime]["position_rmse"])
                        other = (
                            row["intervention"][regime]
                            if baseline == "train_mean_gate"
                            else index[(domain, block, seed, baseline)]["metrics"][
                                regime
                            ]
                        )
                        reference.append(other["position_rmse"])
                    c, r = (
                        sum(candidate) / len(candidate),
                        sum(reference) / len(reference),
                    )
                    gains.append((r - c) / r)
                    ratios.append(math.log(r / c))
                comparisons.append(
                    {
                        "domain": domain,
                        "baseline": baseline,
                        "regime": regime,
                        "block_gains": gains,
                        "mean_gain": sum(gains) / len(gains),
                        "minimum_gain": min(gains),
                        "p_one_sided": sign_permutation(ratios),
                    }
                )
    primary = [
        r
        for r in comparisons
        if r["baseline"] == "radial_fixed" and r["regime"] == "standard"
    ]
    for row, adjusted in zip(
        primary, holm([r["p_one_sided"] for r in primary]), strict=True
    ):
        row["p_holm"] = adjusted
    benefit = all(
        r["mean_gain"] >= protocol["minimum_adaptive_improvement"]
        and r["minimum_gain"] > 0
        and r["p_holm"] <= protocol["alpha"]
        for r in primary
    )
    mechanism = benefit and all(
        r["mean_gain"] >= protocol["minimum_adaptive_improvement"]
        and r["minimum_gain"] > 0
        for r in comparisons
        if r["baseline"] == "train_mean_gate" and r["regime"] == "standard"
    )
    baseline_pass = all(
        r["mean_gain"] >= protocol["minimum_baseline_improvement"]
        and r["minimum_gain"] > 0
        for r in comparisons
        if r["baseline"] in ("graph", "simple_pool", "transformer")
        and r["regime"] == "standard"
    )
    ood = all(
        r["minimum_gain"] >= -protocol["maximum_ood_regression"]
        for r in comparisons
        if r["regime"] != "standard" and r["baseline"] != "train_mean_gate"
    )
    return {
        "decision": "ADAPTIVE_BENEFIT_SUPPORTED_IN_THIS_STUDY"
        if benefit
        else "ADAPTIVE_BENEFIT_NOT_ESTABLISHED",
        "completed": len(rows),
        "required": len(expected),
        "comparisons": comparisons,
        "scene_dependent_mechanism_supported": mechanism,
        "strong_baseline_gate": baseline_pass,
        "ood_gate": ood,
        "independent_blocks_per_domain": len(protocol["data_blocks"]),
        "claim_boundary": "Synthetic prospective successor only; no real-world, SOTA or equivalence claim",
    }
