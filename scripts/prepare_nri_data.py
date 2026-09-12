#!/usr/bin/env python3
"""Generate pinned NRI spring/charge trajectories with a verifiable manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

UPSTREAM_REPOSITORY = "https://github.com/ethanfetaya/NRI.git"
UPSTREAM_COMMIT = "e63fcb0144bca60eb1cffed9a94489de928d6c23"
SPLIT_SEED_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}
DOMAIN_SEED_OFFSETS = {"springs": 0, "charged": 10_000}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_simulators(upstream: Path):
    source = upstream / "data" / "synthetic_sim.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("upstream NRI data/synthetic_sim.py is missing or unsafe")
    spec = importlib.util.spec_from_file_location("ripii_pinned_nri_sim", source)
    if spec is None or spec.loader is None:
        raise ValueError("could not load the pinned NRI simulator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source, module.SpringSim, module.ChargedParticlesSim


def _generate(simulator, scenes: int, horizon: int, sample_freq: int, seed: int):
    if scenes < 1 or horizon < 1 or sample_freq < 1 or seed < 0:
        raise ValueError("generation counts and seed must be positive")
    np.random.seed(seed)
    loc = np.empty((scenes, horizon + 1, 2, simulator.n_balls), dtype=np.float32)
    vel = np.empty_like(loc)
    edges = np.empty((scenes, simulator.n_balls, simulator.n_balls), dtype=np.float32)
    integration_steps = (horizon + 2) * sample_freq
    for index in range(scenes):
        scene_loc, scene_vel, scene_edges = simulator.sample_trajectory(
            T=integration_steps, sample_freq=sample_freq
        )
        if scene_loc.shape != loc.shape[1:] or scene_vel.shape != vel.shape[1:]:
            raise ValueError(
                "pinned NRI simulator returned an unexpected trajectory shape"
            )
        loc[index], vel[index], edges[index] = scene_loc, scene_vel, scene_edges
    if (
        not np.isfinite(loc).all()
        or not np.isfinite(vel).all()
        or not np.isfinite(edges).all()
    ):
        raise FloatingPointError("NRI generation produced non-finite values")
    return loc, vel, edges


def verify_dataset(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("NRI manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "ripii-nri-data-v1" or not isinstance(
        manifest.get("artifacts"), list
    ):
        raise ValueError("invalid NRI manifest schema")
    declared: set[str] = set()
    for entry in manifest["artifacts"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
            or not isinstance(entry.get("bytes"), int)
        ):
            raise ValueError("invalid NRI artifact entry")
        relative = entry["path"]
        path = root / relative
        if (
            relative in declared
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"NRI artifact verification failed: {relative}")
        declared.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if not declared or actual != declared:
        raise ValueError("NRI artifact set differs from its manifest")
    return {"status": "PASS", "artifacts_verified": len(declared)}


def prepare(
    upstream: Path,
    output: Path,
    train_scenes: int,
    eval_scenes: int,
    train_horizon: int,
    test_horizon: int,
    sample_freq: int,
    seed: int,
) -> dict:
    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"refusing existing output directory: {output}")
    if _git_commit(upstream) != UPSTREAM_COMMIT:
        raise ValueError(f"NRI checkout must be pinned to {UPSTREAM_COMMIT}")
    license_path = upstream / "LICENSE"
    if not license_path.is_file() or license_path.is_symlink():
        raise ValueError("pinned NRI checkout lacks a regular LICENSE file")
    simulator_path, spring_cls, charge_cls = _load_simulators(upstream)
    output.mkdir(parents=True)
    artifacts = []
    split_settings = {
        "train": (train_scenes, train_horizon),
        "validation": (eval_scenes, train_horizon),
        "test": (eval_scenes, test_horizon),
    }
    for domain, simulator_cls in (("springs", spring_cls), ("charged", charge_cls)):
        for split, (scenes, horizon) in split_settings.items():
            split_seed = seed + DOMAIN_SEED_OFFSETS[domain] + SPLIT_SEED_OFFSETS[split]
            simulator = simulator_cls(n_balls=5)
            loc, vel, edges = _generate(
                simulator, scenes, horizon, sample_freq, split_seed
            )
            path = output / f"{domain}_{split}.npz"
            np.savez_compressed(
                path,
                loc=loc,
                vel=vel,
                edges=edges,
                seed=np.asarray(split_seed, dtype=np.int64),
            )
            artifacts.append(
                {
                    "path": path.name,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "format": "ripii-nri-data-v1",
        "evidence_class": "external_simulator_development_data",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "license": "MIT",
            "license_sha256": sha256(license_path),
            "simulator_sha256": sha256(simulator_path),
        },
        "domains": ["springs", "charged"],
        "split_policy": (
            "train, validation, and test are generated with disjoint deterministic "
            "NumPy RNG domains; test trajectories are never used for fitting or selection"
        ),
        "generation": {
            "base_seed": seed,
            "split_seed_offsets": SPLIT_SEED_OFFSETS,
            "domain_seed_offsets": DOMAIN_SEED_OFFSETS,
            "train_scenes": train_scenes,
            "eval_scenes": eval_scenes,
            "train_horizon": train_horizon,
            "test_horizon": test_horizon,
            "sample_freq": sample_freq,
            "integration_dt": 0.001,
            "observation_dt": 0.001 * sample_freq,
            "objects": 5,
            "box_size": 5.0,
        },
        "preprocessing": (
            "positions and velocities are divided by the simulator's fixed box_size=5; "
            "no statistic is estimated from train, validation, or test data"
        ),
        "limitations": [
            "locally generated trajectories from an external simulator, not fixed observations",
            "object states are observed directly; perception is out of scope",
            "all scenes contain exactly five objects",
        ],
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verify_dataset(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=128)
    parser.add_argument("--eval-scenes", type=int, default=32)
    parser.add_argument("--train-horizon", type=int, default=16)
    parser.add_argument("--test-horizon", type=int, default=32)
    parser.add_argument("--sample-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=41_003)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_dataset(args.output), indent=2))
        return
    if args.upstream is None:
        raise SystemExit("--upstream is required unless --verify is used")
    result = prepare(
        args.upstream,
        args.output,
        args.train_scenes,
        args.eval_scenes,
        args.train_horizon,
        args.test_horizon,
        args.sample_freq,
        args.seed,
    )
    print(
        json.dumps({"status": "PASS", "artifacts": len(result["artifacts"])}, indent=2)
    )


if __name__ == "__main__":
    main()
