#!/usr/bin/env python3
"""Create a deterministic, complete archive of one retained research run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def collect(root: Path) -> list[tuple[Path, str, bytes]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("run must be a regular directory")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archives reject symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            rows.append((path, relative, path.read_bytes()))
    if not rows:
        raise ValueError("run directory is empty")
    return rows


def create_archive(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite archive")
    if root == output.parent or root in output.parents:
        raise ValueError("archive output must be outside the archived run")
    files = collect(root)
    release_manifest = {
        "format": "ripii-complete-research-archive-v1",
        "root_name": root.name,
        "claim_boundary": (
            "complete byte archive of the selected directory; scientific validity "
            "still depends on protocol and manifest verification"
        ),
        "artifacts": [
            {
                "relative_path": relative,
                "sha256": sha256_bytes(content),
                "size": len(content),
            }
            for _, relative, content in files
        ],
    }
    manifest_bytes = (
        json.dumps(release_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        members = [(relative, content) for _, relative, content in files]
        members.append(("RELEASE_MANIFEST.json", manifest_bytes))
        for relative, content in members:
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    result = {
        "format": "ripii-archive-sidecar-v1",
        "archive": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "size": output.stat().st_size,
        "files": len(files),
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_archive(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
