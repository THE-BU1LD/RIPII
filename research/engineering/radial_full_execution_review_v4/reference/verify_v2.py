"""Verify the immutable v2 protocol and preserved v1 freeze without imports."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    frozen = json.loads((ROOT / "FROZEN.json").read_text())
    for relative, expected in frozen["files"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if protocol["study"] != frozen["study"]:
        raise RuntimeError("Study identity mismatch")
    if protocol["cloud_spend_usd"] != 0:
        raise RuntimeError("Paid execution prohibited")
    if (
        protocol["host_wall_seconds_cap"]
        * protocol["charged_device_count"]
        > protocol["free_gpu_hours_cap"] * 3600
    ):
        raise RuntimeError("Allocated-device budget is internally inconsistent")
    print(json.dumps({"status": "PASS", "files": len(frozen["files"])}))


if __name__ == "__main__":
    main()
