from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ripii.world.data import DatasetSpec, load_dataset
from ripii.world.experiment import load_model, write_json
from ripii.world.physics import Physics
from ripii.world.profiling import ProfileConfig, profile_rollout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a retained RIPII rollout")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--global-coupling", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing profile output: {args.output}")
    model, checkpoint = load_model(args.checkpoint)
    spec = DatasetSpec(
        args.split,
        args.scenes,
        args.horizon,
        args.seed,
        args.max_objects,
        Physics(global_coupling=args.global_coupling),
    )
    data, dataset_record = load_dataset(spec)
    result = profile_rollout(
        model,
        data,
        ProfileConfig(args.warmup, args.repeats, not args.skip_flops),
    )
    result.update(
        {
            "evidence_status": "descriptive_machine_local_profile",
            "checkpoint": {
                "sha256": _sha256(args.checkpoint),
                "selected_step": checkpoint["completed_steps"],
                "model_spec": checkpoint["model_spec"],
                "bytes": args.checkpoint.stat().st_size,
            },
            "dataset": dataset_record,
            "source_sha256": {
                "scripts/profile_world.py": _sha256(Path(__file__)),
                "ripii/world/profiling.py": _sha256(
                    Path(__file__).resolve().parents[1] / "ripii/world/profiling.py"
                ),
            },
            "claim_boundary": (
                "Machine-local descriptive timing; compare models only under an "
                "identical recorded environment and profiling configuration."
            ),
        }
    )
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
