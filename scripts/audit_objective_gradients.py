from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ripii.models.factory import build_model
from ripii.utils.config import load_config
from ripii.utils.objective_diagnostics import objective_gradient_diagnostics
from ripii.utils.seed import seed_everything
from ripii.utils.training import build_data_splits, load_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_output(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("objective diagnostic must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = _signature(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("objective diagnostic is invalid JSON") from exc
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (
        payload.get("format") != "ripii-objective-gradient-audit-v1"
        or payload.get("signature") != expected
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("objective diagnostic signature or schema is invalid")
    for row in rows:
        terms = row.get("diagnostic", {}).get("terms", {})
        if not terms or any(
            not math.isfinite(term.get("weighted_gradient_l2", float("nan")))
            for term in terms.values()
        ):
            raise ValueError("objective diagnostic contains non-finite gradients")
    return {
        "status": "PASS",
        "signature": payload["signature"],
        "seeds_verified": len(rows),
    }


def analyze(run_root: Path, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        run_dir = run_root / "base" / f"seed_{seed}"
        config_path = run_dir / "config.yaml"
        checkpoint_path = run_dir / "final.pt"
        cfg = load_config(config_path)
        seed_everything(cfg.seed)
        train_loader, _, _ = build_data_splits(cfg)
        batch = next(iter(train_loader))
        model = build_model(cfg)
        saved = load_checkpoint(checkpoint_path, model, map_location="cpu")
        diagnostic = objective_gradient_diagnostics(
            model, batch, vars(cfg.loss_weights)
        )
        rows.append(
            {
                "seed": seed,
                "checkpoint_step": saved["step"],
                "checkpoint_sha256": _sha256(checkpoint_path),
                "config_sha256": _sha256(config_path),
                "diagnostic": diagnostic,
            }
        )

    terms = rows[0]["diagnostic"]["active_terms"]
    aggregate_terms = {}
    for term in terms:
        raw = [row["diagnostic"]["terms"][term]["raw_value"] for row in rows]
        norms = [
            row["diagnostic"]["terms"][term]["weighted_gradient_l2"]
            for row in rows
        ]
        aggregate_terms[term] = {
            "raw_value_mean": sum(raw) / len(raw),
            "weighted_gradient_l2_mean": sum(norms) / len(norms),
            "weighted_gradient_l2_min": min(norms),
            "weighted_gradient_l2_max": max(norms),
        }
    cosine_keys = rows[0]["diagnostic"]["pairwise_gradient_cosine"]
    aggregate_cosines = {}
    for key in cosine_keys:
        values = [
            row["diagnostic"]["pairwise_gradient_cosine"][key] for row in rows
        ]
        finite = [value for value in values if value is not None]
        aggregate_cosines[key] = {
            "defined_seeds": len(finite),
            "mean": sum(finite) / len(finite) if finite else None,
            "negative_seeds": sum(value < 0 for value in finite),
        }
    result = {
        "format": "ripii-objective-gradient-audit-v1",
        "evidence_status": "development_diagnostic",
        "experimental_unit": "trained seed; one deterministic training batch per seed",
        "seeds": seeds,
        "source_sha256": {
            "scripts/audit_objective_gradients.py": _sha256(Path(__file__)),
            "ripii/utils/objective_diagnostics.py": _sha256(
                Path(__file__).resolve().parents[1]
                / "ripii/utils/objective_diagnostics.py"
            ),
            "ripii/models/ripii.py": _sha256(
                Path(__file__).resolve().parents[1] / "ripii/models/ripii.py"
            ),
        },
        "rows": rows,
        "aggregate_terms": aggregate_terms,
        "aggregate_pairwise_cosines": aggregate_cosines,
        "claim_boundary": (
            "Post-training local diagnostic on three synthetic seeds. It describes "
            "scale and conflict but cannot assign causal responsibility for outcomes."
        ),
    }
    result["signature"] = _signature(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit legacy objective gradients")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        if any(value is not None for value in (args.run_root, args.seeds, args.output)):
            parser.error("--verify-output cannot be combined with analysis arguments")
        print(json.dumps(verify_output(args.verify_output), indent=2))
        return
    if args.run_root is None or not args.seeds or args.output is None:
        parser.error("analysis requires --run-root, --seeds, and --output")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique and nonnegative")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.output}")
    result = analyze(args.run_root, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(verify_output(args.output), indent=2))


if __name__ == "__main__":
    main()
