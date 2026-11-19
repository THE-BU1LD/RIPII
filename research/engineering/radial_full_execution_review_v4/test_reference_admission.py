"""Qualification-control regressions; subprocess results are artificial fixtures.

These tests execute no torch operations, simulations, training, or CUDA kernels.
The production numerical checks remain unexecuted until an authorized GPU run.
"""

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qualify
from readiness import QUALIFICATION_HORIZONS, admit_full, validate_reference_result


def reference_receipt(name="graph"):
    return {
        "model": name,
        "status": "PASS",
        "checkpoint_replay": "PASS",
        "invalid_input": "REJECTED",
        "optimizer_max_abs": 0.0,
        "checks": [
            {
                "horizon": h,
                "prediction_max_abs": 0.0,
                "gradient_max_abs": 0.0,
                "active_gradients": 1,
            }
            for h in QUALIFICATION_HORIZONS
        ],
    }


class ReferenceAdmission(unittest.TestCase):
    def test_legacy_preflight_receipt_cannot_admit_full_execution(self):
        old = {
            "status": "PASS", "source_sha256": "fixture",
            "all_models": 5, "curriculum_horizons": 7,
            "resume_new_process": "PASS", "gate_intervention": "PASS",
            "capture_eager": "PASS", "test_data_opened": False,
        }
        with self.assertRaises(ValueError):
            admit_full("fixture", old)
        for key in ["capture_lr_change", "frozen_reference_equivalence"]:
            partial = dict(old, capture_lr_change="PASS", frozen_reference_equivalence="PASS")
            partial.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                admit_full("fixture", partial)

    def test_reference_receipt_requires_all_horizons_and_real_gradient_coverage(self):
        valid = reference_receipt()
        validate_reference_result("graph", valid)
        mutations = [
            lambda r: r["checks"].pop(),
            lambda r: r["checks"].__setitem__(1, dict(r["checks"][0])),
            lambda r: r["checks"][0].update(active_gradients=0),
            lambda r: r["checks"][0].update(active_gradients=True),
            lambda r: r.update(model="transformer"),
            lambda r: r.update(checkpoint_replay="NOT_RUN"),
            lambda r: r.update(invalid_input="NOT_RUN"),
        ]
        for mutate in mutations:
            row = copy.deepcopy(valid)
            mutate(row)
            with self.assertRaises(ValueError):
                validate_reference_result("graph", row)

    def test_nonfinite_or_invalid_reference_measurements_fail(self):
        for value in [float("nan"), float("inf"), -1, True, "0"]:
            for key in ["prediction_max_abs", "gradient_max_abs", "optimizer_max_abs"]:
                row = reference_receipt()
                target = row if key == "optimizer_max_abs" else row["checks"][0]
                target[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_reference_result("graph", row)

    def _run_mocked_preflight(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "SOURCE.json").write_text("{}")
            out = root / "out"
            events = []

            def fake_child(command, **kwargs):
                # The mocked child is only a control-flow fixture. It supplies
                # no numerical evidence and never marks a real study qualified.
                if "--mode" not in command:
                    events.append("checkpoint_replay")
                    return subprocess.CompletedProcess(command, 0)
                mode = command[command.index("--mode") + 1]
                name = command[command.index("--name") + 1]
                child = Path(command[command.index("--output") + 1])
                events.append(mode)
                if mode == "reference":
                    if failure == "child":
                        raise subprocess.CalledProcessError(1, command)
                    row = reference_receipt(name)
                    if failure == "receipt":
                        row["checks"].pop()
                    (child / "result.json").write_text(json.dumps(row))
                if mode == "interrupt":
                    (child / "fits").mkdir()
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(qualify, "ROOT", root),
                mock.patch.object(qualify, "NAMES", ("graph",)),
                mock.patch.object(sys, "argv", ["qualify.py", "--output", str(out)]),
                mock.patch.object(qualify.subprocess, "run", side_effect=fake_child),
                mock.patch.dict(os.environ, os.environ.copy(), clear=True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if failure:
                    with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                        qualify.main()
                    self.assertFalse((out / "qualification.json").exists())
                    self.assertEqual(events, ["reference"])
                else:
                    qualify.main()
                    receipt = json.loads((out / "qualification.json").read_text())
                    self.assertEqual(receipt["frozen_reference_equivalence"], "PASS")
                    self.assertEqual(receipt["capture_lr_change"], "PASS")
                    self.assertEqual(
                        events,
                        ["reference", "capture", "continuous", "interrupt", "resume", "checkpoint_replay"],
                    )

    def test_preflight_runs_reference_check_before_capture_and_resume(self):
        self._run_mocked_preflight()

    def test_failed_reference_process_cannot_write_preflight_pass(self):
        self._run_mocked_preflight("child")

    def test_incomplete_reference_receipt_cannot_write_preflight_pass(self):
        self._run_mocked_preflight("receipt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
