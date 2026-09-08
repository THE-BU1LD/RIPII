from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a RIPII artifact manifest and report source drift."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", default="")
    parser.add_argument("--strict-current-source", action="store_true")
    parser.add_argument(
        "--portable-summary",
        action="store_true",
        help=(
            "Verify the protocol and committed summary.json/summary.csv/summary.md "
            "while explicitly omitting manifest entries under runs/. Use the default "
            "full mode when the retained local run directory is available."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    failures: list[str] = []
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit("manifest must be a regular non-symlink file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit("manifest artifacts must be a list")

    artifact_root = manifest_path.parent.resolve()
    seen: set[str] = set()
    verified_paths: set[str] = set()
    verified = 0
    skipped_run_artifacts = 0
    summary_path: Path | None = None
    for entry in artifacts:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("relative_path"), str
        ):
            failures.append("invalid artifact entry")
            continue
        relative = entry["relative_path"]
        if relative in seen:
            failures.append(f"duplicate artifact path: {relative}")
            continue
        seen.add(relative)
        relative_parts = PurePosixPath(relative).parts
        if not relative_parts or PurePosixPath(relative).is_absolute():
            failures.append(f"invalid artifact path: {relative}")
            continue
        path = (artifact_root / relative).resolve()
        if artifact_root not in path.parents:
            failures.append(f"artifact escapes manifest directory: {relative}")
            continue
        if args.portable_summary and relative_parts[0] == "runs":
            skipped_run_artifacts += 1
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"artifact is missing or not a regular file: {relative}")
            continue
        if sha256(path) != entry.get("sha256"):
            failures.append(f"SHA-256 mismatch: {relative}")
            continue
        if path.stat().st_size != entry.get("size"):
            failures.append(f"size mismatch: {relative}")
            continue
        verified += 1
        verified_paths.add(relative)
        if relative == "summary.json":
            summary_path = path

    if args.portable_summary:
        required_summaries = {"summary.json", "summary.csv", "summary.md"}
        for relative in sorted(required_summaries - verified_paths):
            failures.append(f"portable summary artifact did not verify: {relative}")

    if args.protocol:
        protocol_path = Path(args.protocol)
        if protocol_path.is_symlink() or not protocol_path.is_file():
            failures.append("protocol is missing or not a regular file")
        elif sha256(protocol_path) != manifest.get("protocol_sha256"):
            failures.append("protocol SHA-256 mismatch")

    source_drift: list[str] = []
    source_additions: list[str] = []
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_sources = summary.get("source_sha256", {})
        if isinstance(expected_sources, dict):
            for relative, expected in expected_sources.items():
                current = ROOT / relative
                if not current.is_file() or sha256(current) != expected:
                    source_drift.append(relative)
            current_sources = {
                str(path.relative_to(ROOT))
                for base in (ROOT / "ripii", ROOT / "scripts")
                for path in base.rglob("*.py")
            }
            current_sources.add("pyproject.toml")
            source_additions = sorted(current_sources - set(expected_sources))
    if args.strict_current_source and (source_drift or source_additions):
        failures.append("current source differs from recorded run source")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "verification_scope": (
            "portable_summary_only" if args.portable_summary else "full_manifest"
        ),
        "artifacts_verified": verified,
        "artifacts_declared": len(artifacts),
        "run_artifacts_explicitly_skipped": skipped_run_artifacts,
        "source_drift": source_drift,
        "source_additions": source_additions,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
