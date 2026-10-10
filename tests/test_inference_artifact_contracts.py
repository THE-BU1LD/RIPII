"""Constructed archives and concurrent writers; no research datasets are used."""

import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from ripii.world.inference import load_inference_npz, save_prediction_npz


def test_duplicate_archive_members_are_rejected(tmp_path: Path):
    path = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in (
            ("state", np.zeros((1, 8, 6), dtype=np.float32)),
            ("action", np.zeros((1, 8, 2), dtype=np.float32)),
            ("mask", np.ones((1, 8), dtype=np.bool_)),
            ("state", np.ones((1, 8, 6), dtype=np.float32)),
        ):
            buffer = BytesIO()
            np.save(buffer, value, allow_pickle=False)
            warning = pytest.warns(UserWarning) if name == "state" and value.any() else nullcontext()
            with warning:
                archive.writestr(name + ".npy", buffer.getvalue())
    with pytest.raises(ValueError, match="invalid inference NPZ"):
        load_inference_npz(path)


def test_npy_disguised_as_npz_is_a_contract_error(tmp_path: Path):
    path = tmp_path / "not-an-archive.npz"
    with path.open("wb") as handle:
        np.save(handle, np.zeros(2), allow_pickle=False)
    with pytest.raises(ValueError, match="invalid inference NPZ"):
        load_inference_npz(path)


@pytest.mark.parametrize("kind", ["regular", "symlink", "dangling_symlink"])
def test_existing_output_is_preserved(tmp_path: Path, kind):
    path, target = tmp_path / "prediction.npz", tmp_path / "existing.bin"
    original = b"retained original bytes"
    if kind == "regular":
        path.write_bytes(original)
    else:
        if kind == "symlink":
            target.write_bytes(original)
        path.symlink_to(target.name)
    with pytest.raises(FileExistsError):
        save_prediction_npz(path, torch.zeros(1, 8, 6))
    if kind == "regular":
        assert path.read_bytes() == original
    else:
        assert path.is_symlink()
        if kind == "symlink":
            assert target.read_bytes() == original
        else:
            assert not target.exists()


def test_legacy_temporary_symlink_cannot_overwrite_another_file(tmp_path: Path):
    path = tmp_path / "prediction.npz"
    target = tmp_path / "retained.bin"
    target.write_bytes(b"preserve me")
    (tmp_path / ".prediction.npz.tmp").symlink_to(target.name)
    save_prediction_npz(path, torch.ones(1, 8, 6))
    assert target.read_bytes() == b"preserve me"
    assert path.is_file() and not path.is_symlink()
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["states"], np.ones((1, 8, 6)))


def test_concurrent_writers_publish_exactly_one_complete_archive(tmp_path: Path):
    path = tmp_path / "prediction.npz"

    def publish(value):
        try:
            save_prediction_npz(path, torch.full((2, 8, 6), float(value)))
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(publish, range(4)))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["states"], np.full((2, 8, 6), winners[0]))
    assert list(tmp_path.iterdir()) == [path]


def test_failed_serialization_preserves_existing_output_and_cleans_own_temporary(tmp_path, monkeypatch):
    path = tmp_path / "prediction.npz"

    def fail(handle, **kwargs):
        handle.write(b"partial")
        raise OSError("synthetic serialization failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="synthetic serialization failure"):
        save_prediction_npz(path, torch.zeros(1, 8, 6))
    assert not list(tmp_path.iterdir())
