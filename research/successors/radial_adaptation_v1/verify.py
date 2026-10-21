"""Independent artifact and numeric replay, plus safe hosted-archive intake."""

import argparse
import json
import math
import stat
import zipfile
from pathlib import Path

from control import adequate, read, sha, summarize


def verify(run, numeric=False):
    run = Path(run)
    study = run.parent.parent
    manifest = read(run / "manifest.json")["files"]
    actual = {
        str(p.relative_to(run))
        for p in run.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    if actual != set(manifest):
        raise ValueError("Artifact inventory differs")
    for name, digest in manifest.items():
        rel = Path(name)
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or (run / rel).is_symlink()
            or sha(run / rel) != digest
        ):
            raise ValueError(f"Artifact identity differs: {name}")
    frozen = read(study / "FROZEN.json")
    for name, digest in frozen["files"].items():
        if sha(study / name) != digest:
            raise ValueError(f"Source identity differs: {name}")
    header = read(run / "header.json")
    if header["source_sha256"] != sha(study / "FROZEN.json") or header[
        "protocol_sha256"
    ] != sha(study / "protocol.json"):
        raise ValueError("Header identity differs")
    p = read(study / "protocol.json")
    charge = read(run / "charge.json")
    if (
        not math.isfinite(charge["charged_host_seconds"])
        or charge["charged_host_seconds"] < 0
    ):
        raise ValueError("Invalid charge")
    if (
        charge["charged_gpu_seconds"]
        != charge["charged_host_seconds"] * p["charged_device_count"]
    ):
        raise ValueError("GPU charge differs")
    status = read(run / "status.json")
    test_files = list((run / "data").glob("*-test-*.npz"))
    if bool(test_files) != status["test_data_opened"]:
        raise ValueError("Test-access record differs")
    if test_files:
        admission = read(run / "test-admission.json")
        if admission["selections_sha256"] != sha(run / "selections.json"):
            raise ValueError("Test selection lock differs")
    for path in sorted((run / "fits").rglob("receipt.json")):
        row = read(path)
        if row["adequacy"] != adequate(
            row["curve"], row["selected"]["train_rmse"], row["cv_train_rmse"], p
        ):
            raise ValueError("Adequacy replay differs")
        if sha(run / row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs")
        saved = [read(f) for f in sorted(path.parent.glob("curve-*.json"))]
        if saved != row["curve"]:
            raise ValueError("Learning curve differs")
    rows = [read(f) for f in sorted((run / "evaluations").glob("*.json"))]
    if status["status"] == "COMPLETE":
        expected = (
            len(p["domains"])
            * len(p["data_blocks"])
            * len(p["model_seeds"])
            * len(p["models"])
        )
        if len(rows) != expected:
            raise ValueError("Completed run has incomplete matrix")
        if summarize(rows, p) != read(run / "summary.json"):
            raise ValueError("Summary replay differs")
    if numeric:
        import numpy as np

        for row in rows:
            d, b, s, n = (row[k] for k in ("domain", "block", "seed", "model"))
            for regime in p["regimes"]:
                path = run / "predictions" / f"{d}-{b}-{s}-{n}-{regime}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    with np.load(
                        run / "data" / f"{d}-{b}-test-{regime}.npz", allow_pickle=False
                    ) as dataset:
                        if not np.array_equal(
                            archive["target"], dataset["states"][:, p["context"] :]
                        ):
                            raise ValueError(
                                "Prediction target differs from locked dataset"
                            )
                    for key, record in [("prediction", row["metrics"][regime])] + (
                        [("intervention", row["intervention"][regime])]
                        if n == "radial_mixture"
                        else []
                    ):
                        error = archive[key].astype(np.float64) - archive[
                            "target"
                        ].astype(np.float64)
                        if not np.isfinite(error).all():
                            raise ValueError("Nonfinite raw error")
                        scalar = float(np.sqrt(np.mean(error[..., :2] ** 2)))
                        units = np.mean(error[..., :2] ** 2, axis=(1, 2, 3)).tolist()
                        if (
                            scalar != record["position_rmse"]
                            or units != record["scene_position_mse"]
                        ):
                            raise ValueError("Independent raw metric replay differs")
    return {
        "status": "PASS",
        "files": len(manifest),
        "scientific_result": status["scientific_result"],
        "test_data_opened": bool(test_files),
        "evaluations": len(rows),
        "numeric_replay": numeric,
    }


def intake(archive_path, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Intake destination must be new")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError("Duplicate archive members")
        if sum(i.file_size for i in infos) > 8 * 1024**3:
            raise ValueError("Archive exceeds 8 GiB allowance")
        for info in infos:
            rel = Path(info.filename)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive path")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    results = [
        verify(p.parent) for p in sorted(destination.glob("execution/*/status.json"))
    ]
    if not results:
        raise ValueError("No attempt receipts in archive")
    return {"archive_sha256": sha(archive_path), "attempts": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    result = (
        intake(args.path, args.extract)
        if args.extract
        else verify(args.path, args.numeric)
    )
    print(json.dumps(result, indent=2))
