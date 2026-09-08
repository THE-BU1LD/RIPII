from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (".pytest_cache", ".ruff_cache", "build", "dist")
SEARCH_ROOTS = ("ripii", "scripts", "tests")


def generated_targets(root: Path) -> list[Path]:
    root = root.resolve()
    targets = [root / name for name in TOP_LEVEL if (root / name).exists()]
    for relative in SEARCH_ROOTS:
        base = root / relative
        if not base.is_dir():
            continue
        targets.extend(path for path in base.rglob("__pycache__") if path.is_dir())
        targets.extend(path for path in base.rglob("*.pyc") if path.is_file())
    unique = {path.resolve(): path for path in targets if not path.is_symlink()}
    return sorted(unique, key=lambda path: (len(path.parts), str(path)), reverse=True)


def clean(root: Path, *, apply: bool) -> list[Path]:
    root = root.resolve()
    targets = generated_targets(root)
    for target in targets:
        resolved = target.resolve()
        if root not in resolved.parents or "runs" in resolved.relative_to(root).parts:
            raise ValueError(f"refusing unsafe cleanup target: {target}")
        if apply:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            elif resolved.is_file():
                resolved.unlink()
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or remove only disposable RIPII build/test caches; runs are excluded."
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform cleanup (default is dry-run)"
    )
    args = parser.parse_args()
    targets = clean(ROOT, apply=args.apply)
    mode = "removed" if args.apply else "would remove"
    for target in targets:
        print(f"{mode}: {target.relative_to(ROOT)}")
    print(f"{mode} {len(targets)} generated targets; runs/ was never eligible")


if __name__ == "__main__":
    main()
