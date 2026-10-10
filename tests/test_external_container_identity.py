"""The declared dataset root and NPZ members must have unambiguous identities."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
import torch

from ripii.world import external_data
from ripii.world.external_data import load_trajectory_split, verify_trajectory_dataset
from test_external_trajectory_data import build_dataset


def rewrite_manifest(root: Path, split: str, relative: str):
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / relative
    manifest["artifacts"][split] = {"path": relative, "bytes": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("outside", [False, True])
def test_parent_symlink_is_rejected_even_with_valid_size_and_hash(tmp_path, outside):
    root = build_dataset(tmp_path / "dataset")
    target = tmp_path / "external" if outside else root / "nested"
    target.mkdir()
    shutil.copyfile(root / "test.npz", target / "test.npz")
    (root / "link").symlink_to(target, target_is_directory=True)
    rewrite_manifest(root, "test", "link/test.npz")
    with pytest.raises(ValueError, match="failed verification"):
        verify_trajectory_dataset(root)


def test_real_nested_artifact_remains_supported(tmp_path):
    root = build_dataset(tmp_path / "dataset")
    (root / "nested").mkdir()
    shutil.move(root / "test.npz", root / "nested/test.npz")
    rewrite_manifest(root, "test", "nested/test.npz")
    assert verify_trajectory_dataset(root)["status"] == "PASS"


def test_duplicate_npz_member_is_rejected_instead_of_selecting_one_copy(tmp_path):
    root = build_dataset(tmp_path / "dataset")
    path = root / "test.npz"
    with zipfile.ZipFile(path) as archive:
        states = archive.read("states.npy")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("states.npy", states)
    rewrite_manifest(root, "test", "test.npz")
    with pytest.raises(ValueError, match="invalid"):
        load_trajectory_split(root, "test")


def test_verified_artifact_bytes_are_the_bytes_loaded_after_path_replacement(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "dataset")
    expected, expected_record = load_trajectory_split(root, "test")
    replacement = (root / "train.npz").read_bytes()
    original_load = external_data.np.load

    def replace_then_load(source, *args, **kwargs):
        (root / "test.npz").write_bytes(replacement)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(external_data.np, "load", replace_then_load)
    actual, record = load_trajectory_split(root, "test")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert record == expected_record
    assert hashlib.sha256(replacement).hexdigest() != record["source_sha256"]


def test_manifest_digest_describes_the_manifest_actually_parsed(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "dataset")
    manifest_path = root / "manifest.json"
    original_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    changed = json.loads(manifest_path.read_text())
    changed["version"] = "changed-after-parse"
    original_load = external_data.np.load

    def replace_then_load(source, *args, **kwargs):
        manifest_path.write_text(json.dumps(changed))
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(external_data.np, "load", replace_then_load)
    _, record = load_trajectory_split(root, "test")
    assert record["version"] == "1"
    assert record["manifest_sha256"] == original_digest


def test_dataset_verification_uses_one_manifest_snapshot_for_every_split(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "dataset")
    manifest_path = root / "manifest.json"
    original_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    changed = json.loads(manifest_path.read_text())
    changed["version"] = "changed-after-first-split"
    original_load = external_data.np.load

    def replace_then_load(source, *args, **kwargs):
        manifest_path.write_text(json.dumps(changed))
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(external_data.np, "load", replace_then_load)
    result = verify_trajectory_dataset(root)
    assert result["version"] == "1"
    assert result["manifest_sha256"] == original_digest
    for record in result["splits"].values():
        assert record["version"] == result["version"]
        assert record["manifest_sha256"] == original_digest
