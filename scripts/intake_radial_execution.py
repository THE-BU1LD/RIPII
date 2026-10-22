"""Bind an engineering archive to its prelaunch source and cumulative budget."""

import argparse
import hashlib
import json
import math
import stat
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intake(archive, destination, expected_source):
    if destination.exists():
        raise ValueError("Destination already exists")
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError("Duplicate names")
        if sum(i.file_size for i in infos) > 100 * 1024**2:
            raise ValueError("Archive too large")
        for info in infos:
            path = Path(info.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("Unsafe path")
        source.extractall(destination)
    manifest = json.loads((destination / "MANIFEST.json").read_text())
    actual = {
        str(p.relative_to(destination))
        for p in destination.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json"
    }
    if set(manifest) != actual:
        raise ValueError("Manifest inventory differs")
    for name, expected in manifest.items():
        if digest(destination / name) != expected:
            raise ValueError("Artifact changed: " + name)
    local = json.loads(expected_source.read_text())
    remote = json.loads((destination / "SOURCE.json").read_text())
    if local != remote:
        raise ValueError("Unexpected source identity")
    for name, expected in local.items():
        if digest(destination / name) != expected:
            raise ValueError("Source changed: " + name)
    charge = json.loads((destination / "charge.json").read_text())
    for field in (
        "cumulative_charged_gpu_seconds",
        "prior_charged_gpu_seconds",
        "charged_gpu_seconds",
        "elapsed_host_seconds",
        "reserved_export_seconds",
    ):
        component = charge.get(field)
        if (
            type(component) not in (int, float)
            or not 0 <= component <= 14400
            or not math.isfinite(component)
        ):
            raise ValueError("Budget exceeded or invalid component: " + field)
    value = charge["cumulative_charged_gpu_seconds"]
    if value != charge["prior_charged_gpu_seconds"] + charge["charged_gpu_seconds"]:
        raise ValueError("Cumulative charge mismatch")
    if charge["charged_gpu_seconds"] != 2 * (
        charge["elapsed_host_seconds"] + charge["reserved_export_seconds"]
    ):
        raise ValueError("Allocation charge mismatch")
    receipt = {
        "archive_sha256": digest(archive),
        "manifest_files": len(manifest),
        "exact_prelaunch_source_match": True,
        "charge": charge,
        "status": "PASS",
    }
    summary_path = destination / "results/summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        receipt["summary"] = summary
        if summary["engineering_status"] == "PASS":
            rows = json.loads((destination / "results/probe.json").read_text())
            if {r["model"] for r in rows} != {
                "radial_mixture",
                "radial_fixed",
                "graph",
                "simple_pool",
                "transformer",
            } or len(rows) != 5:
                raise ValueError("Incomplete model inventory")
            if not all(r["status"] == "PASS" for r in rows):
                raise ValueError("False success summary")
            seconds = sum(r["profile"]["captured_update_seconds_max"] for r in rows)
            if summary["minimum_update_only_host_seconds"] != 28 * 6000 * seconds:
                raise ValueError("Projection mismatch")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("expected_source", type=Path)
    args = parser.parse_args()
    receipt = intake(args.archive, args.destination, args.expected_source)
    (args.destination.parent / "intake.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))

