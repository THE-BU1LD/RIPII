"""Regression contracts for v3; no CUDA or numerical-library imports."""

import json
import os
import subprocess
import sys
import unittest

from readiness import DEFAULT_SESSION_SECONDS, session_reservation, worker_environment


class RevisionContracts(unittest.TestCase):
    def test_full_session_preserves_one_30_minute_resume(self):
        preflight = session_reservation(3794.580382916, 1800, "PREFLIGHT")
        full = session_reservation(
            preflight["cumulative_ceiling_gpu_seconds"],
            DEFAULT_SESSION_SECONDS["DEVELOPMENT_FULL"],
            "DEVELOPMENT_FULL",
        )
        self.assertAlmostEqual(full["cumulative_ceiling_gpu_seconds"], 58994.580382916)
        self.assertEqual(full["protected_resume_gpu_seconds"], 5400)
        resumed = session_reservation(
            full["cumulative_ceiling_gpu_seconds"], 1800, "RESUME"
        )
        self.assertAlmostEqual(
            resumed["cumulative_ceiling_gpu_seconds"], 64394.580382916
        )
        self.assertEqual(
            resumed["cumulative_ceiling_gpu_seconds"],
            full["cumulative_with_reserved_resume_gpu_seconds"],
        )
        self.assertLess(resumed["cumulative_ceiling_gpu_seconds"], 64800)

    def test_old_full_duration_cannot_consume_recovery_headroom(self):
        preflight = session_reservation(3794.580382916, 1800, "PREFLIGHT")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            session_reservation(
                preflight["cumulative_ceiling_gpu_seconds"], 26000, "DEVELOPMENT_FULL"
            )

    def test_charges_stay_cumulative_across_attempts(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            session_reservation(64394.580382916, 120, "RESUME")

    def test_child_gets_reference_environment_before_imports(self):
        parent = dict(os.environ, CUBLAS_WORKSPACE_CONFIG="incorrect")
        env = worker_environment(parent)
        keys = [
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        ]
        script = (
            "import os,json; print(json.dumps({k:os.environ[k] for k in "
            + repr(keys) + "}))"
        )
        actual = json.loads(subprocess.check_output(
            [sys.executable, "-c", script], env=env, text=True, timeout=10
        ))
        self.assertEqual(actual, {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        })
        self.assertEqual(parent["CUBLAS_WORKSPACE_CONFIG"], "incorrect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
