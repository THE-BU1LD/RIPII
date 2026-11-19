"""Repack a manifest-verified retained run without changing scientific contents."""

import argparse
import json
import zipfile
from pathlib import Path

from readiness import digest, verify_inventory


def pack(root, archive):
    root, archive = Path(root), Path(archive)
    manifest = json.loads((root / "manifest.json").read_text())
    verify_inventory(root, manifest["files"])
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Unregistered or missing results")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name in sorted(expected):
            z.write(root / name, name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        digest(archive) + "  " + archive.name + "\n"
    )
    return {
        "archive": str(archive),
        "sha256": digest(archive),
        "members": len(expected),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("archive")
    a = p.parse_args()
    print(json.dumps(pack(a.root, a.archive)))
