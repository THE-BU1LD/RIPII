from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from ripii.utils.statistics import paired_summary
from ripii.world.experiment import Experiment, load_model, verify, write_json
from ripii.world.failure_analysis import evaluate_failure_regimes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_is_ast_equivalent(current: Path, retained: Path) -> bool:
    """Permit formatting-only drift while rejecting executable source drift."""
    current_tree = ast.parse(
        current.read_text(encoding="utf-8"), filename=str(current), type_comments=True
    )
    retained_tree = ast.parse(
        retained.read_text(encoding="utf-8"), filename=str(retained), type_comments=True
    )
    return ast.dump(current_tree, include_attributes=False) == ast.dump(
        retained_tree, include_attributes=False
    )


def _analysis_signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _all_finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))


def verify_analysis(path: Path) -> dict[str, str | int | bool]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("analysis must be a regular non-symlink file")
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    if not isinstance(payload, dict) or not _all_finite(payload):
        raise ValueError("analysis must be a finite JSON object")
    if payload.get("format") != "ripii-world-failure-analysis-v1":
        raise ValueError("invalid failure-analysis format")
    if payload.get("signature") != _analysis_signature(payload):
        raise ValueError("failure-analysis signature mismatch")
    for key in ("full_run_verification", "post_analysis_run_verification"):
        verification = payload.get(key)
        if (
            not isinstance(verification, dict)
            or verification.get("status") != "PASS"
            or verification.get("unexpected_files") != 0
            or not isinstance(verification.get("artifacts_verified"), int)
            or verification["artifacts_verified"] < 1
        ):
            raise ValueError(f"invalid {key}")
    model_source = payload.get("checkpoint_model_source")
    if not isinstance(model_source, dict) or not (
        model_source.get("exact_match") is True
        or model_source.get("ast_equivalent") is True
    ):
        raise ValueError("checkpoint model source was not authenticated")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("failure analysis requires result rows")
    cells = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("failure-analysis rows must be objects")
        cell = (row.get("variant"), row.get("seed"))
        if (
            not isinstance(cell[0], str)
            or not isinstance(cell[1], int)
            or isinstance(cell[1], bool)
            or cell in cells
        ):
            raise ValueError("invalid or duplicate failure-analysis cell")
        cells.add(cell)
    return {
        "status": "PASS",
        "signature": payload["signature"],
        "rows_verified": len(rows),
        "finite_values": True,
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def analyze(run_dir: Path, *, near_margin: float = 0.05) -> dict:
    run_dir = Path(run_dir)
    verification = verify(run_dir)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    experiment = Experiment(**protocol["experiment"])
    experiment.validate()
    critical_sources = ("ripii/world/models.py", "ripii/world/physics.py")
    root = Path(__file__).resolve().parents[1]
    for relative in critical_sources:
        snapshot = run_dir / "source" / relative
        if _sha256(snapshot) != protocol["source_sha256"].get(relative):
            raise ValueError(f"retained critical source is invalid: {relative}")
    current_models = root / "ripii/world/models.py"
    retained_models = run_dir / "source/ripii/world/models.py"
    model_source_exact = _sha256(current_models) == protocol["source_sha256"].get(
        "ripii/world/models.py"
    )
    model_source_ast_equivalent = _source_is_ast_equivalent(
        current_models, retained_models
    )
    if not model_source_exact and not model_source_ast_equivalent:
        raise ValueError("current model source differs from retained checkpoints")
    physics_path = run_dir / "source/ripii/world/physics.py"
    spec = importlib.util.spec_from_file_location(
        "ripii_retained_physics", physics_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load retained physics source")
    retained_physics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = retained_physics
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(retained_physics)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        sys.modules.pop(spec.name, None)
    variants = [
        name
        for name in ("graph", "global_pool", "multiscale")
        if name in protocol["variants"]
    ]
    seeds = [int(seed) for seed in protocol["seeds"]]
    datasets = {
        split: retained_physics.make_dataset(
            split,
            experiment.eval_scenes,
            experiment.test_horizon,
            experiment.data_seed,
            experiment.max_objects,
            retained_physics.Physics(**protocol["physics"]),
        )
        for split in ("test", "more_objects", "composition", "fast")
    }
    rows = []
    for variant in variants:
        for seed in seeds:
            checkpoint = run_dir / f"{variant}_continuous" / f"seed_{seed}" / "best.pt"
            model, saved = load_model(checkpoint)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint": str(checkpoint.relative_to(run_dir)),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "selected_step": saved["completed_steps"],
                    "splits": {
                        split: evaluate_failure_regimes(
                            model, data, near_margin=near_margin
                        )
                        for split, data in datasets.items()
                    },
                }
            )
    comparisons = {}
    by_key = {(row["variant"], row["seed"]): row for row in rows}
    for baseline in ("graph", "global_pool"):
        if baseline not in variants or "multiscale" not in variants:
            continue
        comparison = {}
        for split in datasets:
            comparison[split] = {}
            for regime in (
                "contact",
                "near_contact",
                "wall",
                "forced",
                "free_flight",
                "all",
            ):
                candidate, control, paired_seeds = [], [], []
                for seed in seeds:
                    a = by_key[("multiscale", seed)]["splits"][split][regime]
                    b = by_key[(baseline, seed)]["splits"][split][regime]
                    if "position_rmse" in a and "position_rmse" in b:
                        candidate.append(a["position_rmse"])
                        control.append(b["position_rmse"])
                        paired_seeds.append(seed)
                comparison[split][regime] = (
                    {
                        "seeds": paired_seeds,
                        **paired_summary(
                            candidate,
                            control,
                            bootstrap_seed=70_000 + len(comparison[split]),
                        ),
                    }
                    if paired_seeds
                    else {"seeds": [], "status": "no_observations"}
                )
        comparisons[baseline] = comparison
    post_analysis_verification = verify(run_dir)
    result = {
        "format": "ripii-world-failure-analysis-v1",
        "evidence_status": "exploratory_development_failure_analysis",
        "source_run": run_dir.name,
        "source_protocol_sha256": _sha256(run_dir / "protocol.json"),
        "full_run_verification": verification,
        "post_analysis_run_verification": post_analysis_verification,
        "analysis_source_sha256": {
            "scripts/analyze_failures.py": _sha256(Path(__file__).resolve()),
            "ripii/world/failure_analysis.py": _sha256(
                root / "ripii/world/failure_analysis.py"
            ),
            "ripii/world/experiment.py": _sha256(
                root / "ripii/world/experiment.py"
            ),
            "ripii/utils/statistics.py": _sha256(
                root / "ripii/utils/statistics.py"
            ),
        },
        "checkpoint_model_source": {
            "current_sha256": _sha256(current_models),
            "retained_sha256": _sha256(retained_models),
            "exact_match": model_source_exact,
            "ast_equivalent": model_source_ast_equivalent,
            "policy": "exact bytes or identical Python AST; executable drift is rejected",
        },
        "near_contact_margin": near_margin,
        "regime_definition": "pre-transition true state; categories overlap except free_flight",
        "experimental_unit": "paired initialization/minibatch seed on one fixed simulator dataset",
        "rows": rows,
        "comparisons": comparisons,
        "claim_boundary": "Post-result exploratory localization. It cannot change or confirm the frozen advancement decision.",
    }
    result["signature"] = _analysis_signature(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize retained world-model rollout failures"
    )
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        if args.run_dir is not None or args.output is not None:
            parser.error("--verify-output cannot be combined with analysis arguments")
        print(json.dumps(verify_analysis(args.verify_output), indent=2))
        return
    if args.run_dir is None or args.output is None:
        parser.error("analysis requires run_dir and --output")
    result = analyze(args.run_dir, near_margin=args.near_margin)
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    write_json(args.output, result)


if __name__ == "__main__":
    main()
