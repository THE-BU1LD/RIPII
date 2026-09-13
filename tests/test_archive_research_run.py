from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path


def _module():
    path = Path("scripts/archive_research_run.py")
    spec = importlib.util.spec_from_file_location("archive_research_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_archive_is_deterministic_and_self_describing(tmp_path):
    module = _module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint.pt").write_bytes(b"checkpoint")
    (run / "metrics.json").write_text("{}\n", encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    one = module.create_archive(run, first)
    two = module.create_archive(run, second)
    assert one["sha256"] == two["sha256"]
    assert json.loads(first.with_suffix(".gz.json").read_text())["files"] == 2
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert names == ["checkpoint.pt", "metrics.json", "RELEASE_MANIFEST.json"]
        manifest = json.load(archive.extractfile("RELEASE_MANIFEST.json"))
    assert len(manifest["artifacts"]) == 2
