"""Budget, identity, archive and frozen-study controls; no tensor imports."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from readiness import admit_full, identity, reservation, safe_relative, verify_inventory


class Controls(unittest.TestCase):
    def test_old_fixed_reservation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            reservation(3794.580382916, 30000)
        self.assertLessEqual(
            reservation(3794.580382916, 29000)["cumulative_ceiling_gpu_seconds"], 64800
        )

    def test_no_nonfinite_or_negative_budget(self):
        for value in [float("nan"), float("inf"), -1, True, "0"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reservation(value, 100)

    def test_fractional_or_zero_devices_rejected(self):
        for devices in [0, 0.5, 1.5, True]:
            with self.assertRaises(ValueError):
                reservation(0, 100, devices=devices)

    def test_boundary_charge(self):
        self.assertEqual(
            reservation(0, 90, 10, 200, 2)["cumulative_ceiling_gpu_seconds"], 200
        )
        with self.assertRaises(ValueError):
            reservation(0.01, 90, 10, 200, 2)

    def test_safe_paths(self):
        for path in ["../x", "/x", "a/../b", "a\\b", "a//b", ""]:
            with self.assertRaises(ValueError):
                safe_relative(path)
        self.assertEqual(str(safe_relative("data/file.json")), "data/file.json")

    def test_inventory_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"
            p.write_bytes(b"initial")
            files = {"x": hashlib.sha256(b"initial").hexdigest()}
            verify_inventory(tmp, files)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_inventory(tmp, files)

    def test_inventory_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"
            p.symlink_to("/etc/hosts")
            with self.assertRaises(ValueError):
                verify_inventory(tmp, {"x": "0" * 64})

    def test_qualification_requires_every_gate_and_exact_source(self):
        row = {
            "status": "PASS",
            "source_sha256": "abc",
            "all_models": 5,
            "curriculum_horizons": 7,
            "resume_new_process": "PASS",
            "gate_intervention": "PASS",
            "capture_eager": "PASS",
            "capture_lr_change": "PASS",
            "frozen_reference_equivalence": "PASS",
            "test_data_opened": False,
        }
        admit_full("abc", row)
        for key in row:
            changed = dict(row)
            changed.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                admit_full("abc", changed)
        with self.assertRaises(ValueError):
            admit_full("different", row)

    def test_identity_independent_of_dict_order(self):
        self.assertEqual(identity({"a": "1", "b": "2"}), identity({"b": "2", "a": "1"}))
        self.assertNotEqual(identity({"a": "1"}), identity({"a": "2"}))

    def test_source_compiles_without_importing_torch(self):
        for p in Path(__file__).parent.glob("*.py"):
            with self.subTest(path=p.name):
                ast.parse(p.read_text(), str(p))

    def test_frozen_predecessor_unchanged(self):
        here = Path(__file__).resolve().parent
        baseline = json.loads((here / "BASELINE_SOURCE_V3.json").read_text())
        # This checkout integrates the complete portable source. The absent
        # original workstation layout is not a dependency of its reproduction.
        verify_inventory(here, {k: v for k, v in baseline.items() if k.startswith("reference/")})
        verify_inventory(here, json.loads((here / "SOURCE.json").read_text()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
