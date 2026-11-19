"""Artifact roundtrip and mutation checks without numerical dependencies."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from aggregate_results import aggregate
from export_contract import emit
from notebook_runtime import run, unpack
from orchestrate import copy_resume_snapshot
from pack_results import pack
from readiness import reservation


def fixture(root):
    (root / "wrapper-status.json").write_text(
        json.dumps({"status": "FAILED", "scientific_result": "NOT_EVALUATED"})
    )
    files = {
        "wrapper-status.json": hashlib.sha256(
            (root / "wrapper-status.json").read_bytes()
        ).hexdigest()
    }
    (root / "manifest.json").write_text(json.dumps({"files": files}))


class Artifacts(unittest.TestCase):
    def test_failed_export_reports_absences_without_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            emit(root, {"status": "FAILED", "reason": "fixture failure"})
            contract = json.loads((root / "artifact_contract.json").read_text())
            self.assertEqual(contract["completed_fit_receipts"], 0)
            self.assertFalse(contract["items"]["metrics.csv"]["exists"])
            self.assertIn("fixture failure", (root / "failures.jsonl").read_text())

    def test_budget_receipt_retains_prior_session_charge(self):
        first = reservation(3794.580382916, 1800)
        second = reservation(first["cumulative_ceiling_gpu_seconds"], 26000)
        self.assertAlmostEqual(
            second["cumulative_ceiling_gpu_seconds"], 62994.580382916
        )

    def test_multiple_timeouts_keep_latest_and_older_unvisited_fits(self):
        with tempfile.TemporaryDirectory() as d:
            previous = Path(d) / "previous"
            for relative, content in [
                ("shards/new/ fits/main/one/latest.pt", "new-one"),
                ("resume/shards/old/ fits/main/one/latest.pt", "old-one"),
                ("resume/resume/shards/old/ fits/main/two/latest.pt", "old-two"),
            ]:
                p = previous / relative.replace("/ fits/", "/fits/")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            destination = Path(d) / "snapshot"
            copy_resume_snapshot(previous, destination)
            fits = sorted(
                p.read_text() for p in destination.glob("shards/*/fits/*/*/latest.pt")
            )
            self.assertEqual(fits, ["new-one", "old-two"])
            self.assertFalse((destination / "resume").exists())

    def test_confirmation_is_refused_before_source_or_data_access(self):
        with self.assertRaisesRegex(RuntimeError, "no independently frozen"):
            run("nonexistent", "CONFIRMATORY_FROZEN", 1800)

    def test_invalid_mode_is_refused(self):
        with self.assertRaises(ValueError):
            run("nonexistent", "UNBOUNDED", 1800)

    def test_session_duration_rejected_before_source_access(self):
        for seconds in (0, -1, True, 10, 1800.5):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                run("nonexistent", "PREFLIGHT", seconds)

    def test_empty_result_is_pending_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            fixture(root)
            result = aggregate(root, Path(d) / "analysis")
            self.assertEqual(result["status"], "PENDING_EXPERIMENTAL_EVIDENCE")
            self.assertEqual(result["metrics"], [])
            self.assertIn("PENDING", (Path(d) / "analysis/results.tex").read_text())

    def test_result_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            fixture(root)
            target = Path(d) / "results.zip"
            receipt = pack(root, target)
            unpack(target, Path(d) / "extracted")
            self.assertEqual(receipt["members"], 2)
            self.assertEqual(
                (root / "wrapper-status.json").read_bytes(),
                (Path(d) / "extracted/wrapper-status.json").read_bytes(),
            )

    def test_corrupt_result_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            fixture(root)
            (root / "wrapper-status.json").write_text("{}")
            with self.assertRaises(ValueError):
                pack(root, Path(d) / "results.zip")

    def test_unregistered_result_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            fixture(root)
            (root / "unexpected.txt").write_text("unregistered")
            with self.assertRaises(ValueError):
                pack(root, Path(d) / "results.zip")

    def test_unregistered_nested_manifest_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            fixture(root)
            target = Path(d) / "results.zip"
            pack(root, target)
            with zipfile.ZipFile(target, "a") as z:
                z.writestr("execution/manifest.json", "{}")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                unpack(target, Path(d) / "extracted")

    def test_unsafe_archives_rejected(self):
        for name in ("../outside", "/absolute", "a\\b"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "bad.zip"
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr(name, "x")
                with self.assertRaises(ValueError):
                    unpack(path, Path(d) / "extract")

    def test_duplicate_archive_members_rejected(self):
        import warnings

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("same", "a")
                    z.writestr("same", "b")
            with self.assertRaises(ValueError):
                unpack(path, Path(d) / "extract")


if __name__ == "__main__":
    unittest.main(verbosity=2)
