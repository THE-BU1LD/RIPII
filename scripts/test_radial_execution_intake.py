import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from intake_radial_execution import intake


class IntakeTests(unittest.TestCase):
    def fixture(self, root, *, wrong_source=False, extra=None, charge=10):
        digest = hashlib.sha256(b"print('fixture')\n").hexdigest()
        source = {"probe.py": digest}
        expected = root / "expected.json"
        expected.write_text(
            json.dumps({"probe.py": "0" * 64} if wrong_source else source)
        )
        files = {
            "probe.py": b"print('fixture')\n",
            "SOURCE.json": json.dumps(source).encode(),
            "charge.json": json.dumps(
                {
                    "cumulative_charged_gpu_seconds": charge,
                    "prior_charged_gpu_seconds": 0,
                    "charged_gpu_seconds": charge,
                    "elapsed_host_seconds": charge / 2,
                    "reserved_export_seconds": 0,
                }
            ).encode(),
        }
        manifest = {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        }
        files["MANIFEST.json"] = json.dumps(manifest).encode()
        archive = root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
            if extra:
                z.writestr(extra, b"unexpected")
        return archive, expected

    def test_binds_expected_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, expected = self.fixture(root)
            self.assertEqual(intake(archive, root / "out", expected)["status"], "PASS")

    def test_rejects_self_consistent_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, expected = self.fixture(root, wrong_source=True)
            with self.assertRaisesRegex(ValueError, "Unexpected source"):
                intake(archive, root / "out", expected)

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, expected = self.fixture(root, extra="../outside")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                intake(archive, root / "out", expected)
            self.assertFalse((root / "outside").exists())

    def test_rejects_unmanifested_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, expected = self.fixture(root, extra="extra.json")
            with self.assertRaisesRegex(ValueError, "inventory"):
                intake(archive, root / "out", expected)

    def test_rejects_over_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, expected = self.fixture(root, charge=14401)
            with self.assertRaisesRegex(ValueError, "Budget"):
                intake(archive, root / "out", expected)


if __name__ == "__main__":
    unittest.main()
