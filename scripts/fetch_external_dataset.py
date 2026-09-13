"""Fetch or verify a registry-pinned external research dataset artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path, PurePath


def load_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_id",
        "version",
        "doi",
        "license",
        "source_url",
        "filename",
        "bytes",
        "checksum_algorithm",
        "checksum",
    }
    if (
        not isinstance(record, dict)
        or record.get("format") != "ripii-external-source-v1"
        or not required <= record.keys()
        or any(not isinstance(record[key], str) or not record[key] for key in required - {"bytes"})
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] < 1
        or record["checksum_algorithm"] not in hashlib.algorithms_available
        or PurePath(record["filename"]).name != record["filename"]
        or not record["source_url"].startswith("https://")
    ):
        raise ValueError("external source record is invalid")
    return record


def verify_file(record: dict, path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("external artifact is missing or unsafe")
    digest = hashlib.new(record["checksum_algorithm"])
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            sha256.update(block)
    if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["checksum"]:
        raise ValueError("external artifact size or publisher checksum mismatch")
    if record.get("retrieved_sha256") not in {None, sha256.hexdigest()}:
        raise ValueError("external artifact retrieved SHA-256 mismatch")
    return {
        "status": "PASS",
        "dataset_id": record["dataset_id"],
        "version": record["version"],
        "license": record["license"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "checksum_algorithm": record["checksum_algorithm"],
        "checksum": digest.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def fetch(record: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / record["filename"]
    if destination.exists():
        return verify_file(record, destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"remove or resume the incomplete download: {temporary}")
    request = urllib.request.Request(
        record["source_url"], headers={"User-Agent": "RIPII-research-fetch/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        verify_file(record, temporary)
        temporary.replace(destination)
    except Exception:
        # Preserve partial bytes for diagnosis; never promote them to the final name.
        raise
    return verify_file(record, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    record = load_record(args.record)
    destination = args.output_dir / record["filename"]
    result = verify_file(record, destination) if args.verify else fetch(record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
