"""Create/verify prospective alpha-aware plans without rewriting retained v1."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from ripii.utils.power_v2 import FORMAT, approximate_paired_seed_count
from scripts.plan_power import _sha256, _signature
from scripts.plan_power import plan as _legacy_plan

ROOT = Path(__file__).resolve().parents[1]
_SOURCES = (
    "scripts/plan_power.py",
    "ripii/utils/power.py",
    "scripts/plan_power_v2.py",
    "ripii/utils/power_v2.py",
)


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(ROOT / name) for name in _SOURCES}


def plan(path: Path, **kwargs) -> dict:
    # Reuse the established summary/capsule admission, pairing and provenance
    # path. Only the declared prospective planning arithmetic changes.
    result = _legacy_plan(path, **kwargs)
    corrected = approximate_paired_seed_count(
        result["development_differences"],
        minimum_detectable_effect=result["minimum_detectable_effect"],
        alpha=result["two_sided_alpha"],
        power=result["target_power"],
    )
    result.update(corrected)
    result["source_sha256"] = _source_hashes()
    result["signature"] = _signature(result)
    return result


def verify_plan_artifact(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("power plan must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError("expected an explicit v2 power plan")
        if payload.get("signature") != _signature(payload):
            raise ValueError("power-plan content signature is invalid")
        expected = approximate_paired_seed_count(
            payload["development_differences"],
            minimum_detectable_effect=payload["minimum_detectable_effect"],
            alpha=payload["two_sided_alpha"],
            power=payload["target_power"],
        )
        if any(
            type(payload.get(key)) is not type(value) or payload[key] != value
            for key, value in expected.items()
        ):
            raise ValueError(
                "power-plan arithmetic does not match its declared development inputs"
            )
        seeds = payload["seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != expected["development_pairs"]
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or seeds != sorted(set(seeds))
        ):
            raise ValueError(
                "power-plan seeds must identify every ordered development pair"
            )
        if (
            payload.get("evidence_status")
            != "prospective_planning_from_development_variance"
        ):
            raise ValueError("power-plan evidence status is invalid")
        if payload.get("source_sha256") != _source_hashes():
            raise ValueError(
                "power-plan source identities differ from this v2 verifier"
            )
    except (KeyError, TypeError, OverflowError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("power plan is invalid or incomplete") from exc
    return {
        "status": "PASS",
        "format": FORMAT,
        "signature": payload["signature"],
        "recommended_minimum_pairs": expected["recommended_minimum_pairs"],
        "exact_sign_flip_resolution_floor_pairs": expected[
            "exact_sign_flip_resolution_floor_pairs"
        ],
    }


def _publish_new(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            staging = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-filesystem hard-link publication is atomic and refuses an existing
        # destination (including a symlink). No retained plan is replaced.
        os.link(staging, path)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a prospective alpha-aware RIPII power plan"
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--candidate", default="multiscale")
    parser.add_argument("--baseline", default="graph")
    parser.add_argument("--split", default="more_objects")
    parser.add_argument("--bottleneck", default="continuous")
    parser.add_argument("--minimum-detectable-effect", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        if args.input is not None or args.output is not None:
            parser.error("--verify-output cannot be combined with planning arguments")
        print(json.dumps(verify_plan_artifact(args.verify_output), indent=2))
        return
    if args.input is None or args.output is None:
        parser.error("planning requires input and --output")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing power-plan output: {args.output}")
    result = plan(
        args.input,
        candidate=args.candidate,
        baseline=args.baseline,
        split=args.split,
        bottleneck=args.bottleneck,
        minimum_detectable_effect=args.minimum_detectable_effect,
        alpha=args.alpha,
        power=args.power,
    )
    _publish_new(args.output, result)
    print(
        json.dumps(
            {"output": str(args.output), **verify_plan_artifact(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
