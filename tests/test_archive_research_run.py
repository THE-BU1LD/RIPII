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


def test_archive_streams_files_instead_of_using_path_read_bytes(tmp_path, monkeypatch):
    module = _module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "large.bin").write_bytes(b"stream-me" * 1024)

    def reject_read_bytes(_path):
        raise AssertionError("archive attempted a whole-file Path.read_bytes call")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    result = module.create_archive(run, tmp_path / "streamed.tar.gz")
    assert result["files"] == 1
