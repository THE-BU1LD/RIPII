from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.fetch_external_dataset import load_record, verify_file


def _record(tmp_path: Path, content: bytes = b"external-data") -> tuple[Path, Path]:
    artifact = tmp_path / "data.zip"
    artifact.write_bytes(content)
    record = {
        "format": "ripii-external-source-v1",
        "dataset_id": "fixture",
        "version": "1",
        "doi": "10.example/fixture",
        "license": "CC-BY-4.0",
        "source_url": "https://example.invalid/data.zip",
        "filename": artifact.name,
        "bytes": len(content),
        "checksum_algorithm": "sha256",
        "checksum": hashlib.sha256(content).hexdigest(),
    }
    source = tmp_path / "record.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    return source, artifact


def test_external_source_record_and_artifact_verify(tmp_path: Path) -> None:
    source, artifact = _record(tmp_path)
    result = verify_file(load_record(source), artifact)
    assert result["status"] == "PASS"
    assert result["bytes"] == len(b"external-data")


def test_external_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    source, artifact = _record(tmp_path)
    artifact.write_bytes(b"tampered-data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_file(load_record(source), artifact)


def test_external_record_rejects_unsafe_filename(tmp_path: Path) -> None:
    source, _ = _record(tmp_path)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["filename"] = "../escape.zip"
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_record(source)
