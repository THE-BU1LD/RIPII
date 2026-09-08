from __future__ import annotations

from scripts.clean_generated import clean


def test_cleanup_is_dry_run_by_default_and_never_touches_runs(tmp_path) -> None:
    cache = tmp_path / "ripii/module/__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"cache")
    evidence = tmp_path / "runs/evidence/__pycache__"
    evidence.mkdir(parents=True)
    (evidence / "retained.pyc").write_bytes(b"evidence")
    targets = clean(tmp_path, apply=False)
    assert cache.resolve() in [target.resolve() for target in targets]
    assert cache.exists()
    clean(tmp_path, apply=True)
    assert not cache.exists()
    assert evidence.exists()
