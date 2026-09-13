"""Fail closed until legal, authorship, and immutable release inputs are complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(root: Path) -> dict:
    required_files = (
        "LICENSE",
        "CITATION.cff",
        "release/OWNER_METADATA.json",
        "output/pdf/ripii-manuscript.pdf",
        "output/release/ripii.cdx.json",
    )
    missing = [name for name in required_files if not (root / name).is_file()]
    invalid = []
    metadata_path = root / "release/OWNER_METADATA.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        required_metadata = {
            "authors",
            "copyright_holder",
            "software_license_spdx",
            "funding",
            "conflicts_of_interest",
            "author_contributions",
            "approved_by",
            "approved_at_utc",
        }
        if (
            not isinstance(metadata, dict)
            or not required_metadata <= metadata.keys()
            or not isinstance(metadata.get("authors"), list)
            or not metadata.get("authors")
            or any(value == "REQUIRED" for value in metadata.values())
        ):
            invalid.append("release/OWNER_METADATA.json")
    result = {
        "status": "PASS" if not missing and not invalid else "BLOCKED",
        "missing": missing,
        "invalid": invalid,
    }
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    try:
        result = check(parser.parse_args().root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
