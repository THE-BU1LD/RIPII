#!/usr/bin/env python3
"""Build a non-authoritative inventory of retained, failed, and partial run folders."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def read_status(directory: Path) -> tuple[str, str | None]:
    for name in ("status.json", "launch.json"):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "broken_metadata", name
        status = payload.get("status")
        if not isinstance(status, str):
            events = payload.get("events", [])
            if isinstance(events, list) and events:
                final = events[-1]
                status = final.get("state") if isinstance(final, dict) else None
        return status if isinstance(status, str) else "unknown", name
    if (directory / "manifest.json").is_file():
        return "manifest_present", "manifest.json"
    if ".incomplete." in directory.name:
        return "quarantined_incomplete", None
    return "unclassified", None


def inventory(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("run root must be a regular directory")
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = [
            item
            for item in directory.rglob("*")
            if item.is_file() and not item.is_symlink()
        ]
        status, source = read_status(directory)
        rows.append(
            {
                "run": directory.name,
                "status": status,
                "status_source": source,
                "files": len(files),
                "bytes": sum(item.stat().st_size for item in files),
                "latest_mtime": max(
                    (item.stat().st_mtime for item in files), default=directory.stat().st_mtime
                ),
            }
        )
    if any(
        not math.isfinite(row["latest_mtime"]) or row["bytes"] < 0 for row in rows
    ):
        raise FloatingPointError("invalid run inventory metadata")
    return {
        "format": "ripii-run-index-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "claim_boundary": (
            "inventory only; a manifest must be cryptographically verified before "
            "a run is treated as valid evidence"
        ),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/index.json"))
    args = parser.parse_args()
    payload = inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"runs": len(payload["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()

