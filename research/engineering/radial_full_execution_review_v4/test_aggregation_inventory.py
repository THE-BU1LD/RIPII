"""Result-inventory regression fixtures; no training or scientific data."""

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from aggregate_results import aggregate


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def manifest(root, names=None):
    selected = (
        [root / name for name in names]
        if names is not None
        else [p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"]
    )
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in selected
    }
    write(root, "manifest.json", json.dumps({"files": files}))


def pending_fixture(root):
    root.mkdir()
    write(root, "wrapper-status.json", '{"status":"FAILED"}')
    manifest(root)


def evaluation_fixture(root, marker=None):
    write(root, "execution/evaluations.json", "[]")
    write(root, "execution/summary.json", '{"decision":"INCOMPLETE"}')
    write(root, "source/reference/protocol.json", "{}")
    source = "def summarize(rows, protocol):\n    return {'decision': 'INCOMPLETE'}\n"
    if marker is not None:
        source = (
            "from pathlib import Path\n"
            + "Path(" + repr(str(marker)) + ").write_text('executed')\n"
            + source
        )
    write(root, "source/reference/control.py", source)


class AggregationInventory(unittest.TestCase):
    def test_unlisted_result_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            output = Path(directory) / "analysis"
            pending_fixture(root)
            write(root, "unexpected.txt", "not in the retained inventory")
            with self.assertRaisesRegex(ValueError, "Unregistered or missing"):
                aggregate(root, output)
            self.assertFalse(output.exists())

    def test_unlisted_control_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            output = Path(directory) / "analysis"
            marker = Path(directory) / "unverified-source-executed"
            pending_fixture(root)
            evaluation_fixture(root, marker)
            with self.assertRaisesRegex(ValueError, "Unregistered or missing"):
                aggregate(root, output)
            self.assertFalse(marker.exists())
            self.assertFalse(output.exists())

    def test_unlisted_prediction_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            pending_fixture(root)
            write(root, "execution/shards/one/predictions/unlisted.npz", "fixture")
            with self.assertRaisesRegex(ValueError, "Unregistered or missing"):
                aggregate(root, Path(directory) / "analysis")

    def test_nested_manifest_is_not_exempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            pending_fixture(root)
            write(root, "execution/manifest.json", "{}")
            with self.assertRaisesRegex(ValueError, "Unregistered or missing"):
                aggregate(root, Path(directory) / "analysis")

    def test_symlinked_directory_cannot_bypass_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            external = Path(directory) / "external"
            external.mkdir()
            write(external, "evaluations.json", "[]")
            pending_fixture(root)
            (root / "execution").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                aggregate(root, Path(directory) / "analysis")

    def test_registered_corruption_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            pending_fixture(root)
            write(root, "wrapper-status.json", '{"status":"COMPLETE"}')
            with self.assertRaisesRegex(ValueError, "Artifact identity differs"):
                aggregate(root, Path(directory) / "analysis")

    def test_registered_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            pending_fixture(root)
            (root / "wrapper-status.json").unlink()
            with self.assertRaisesRegex(ValueError, "Unregistered or missing"):
                aggregate(root, Path(directory) / "analysis")

    def test_verified_pending_evaluation_does_not_modify_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            pending_fixture(root)
            evaluation_fixture(root)
            manifest(root)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            # Empty evaluations do not exercise NumPy. Keep this engineering
            # fixture independent of optional numerical-package installation.
            with mock.patch.dict("sys.modules", {"numpy": types.ModuleType("numpy")}):
                result = aggregate(root, Path(directory) / "analysis")
                repeated = aggregate(root, Path(directory) / "analysis-replay")
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(result, repeated)
            self.assertEqual(result["status"], "PENDING_EXPERIMENTAL_EVIDENCE")
            self.assertFalse(list(root.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
