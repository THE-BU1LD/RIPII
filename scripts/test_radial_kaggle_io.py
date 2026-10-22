"""Local transport guards; every API call is mocked and no workload is launched."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SOURCE = Path(__file__).with_name("radial_kaggle_io.py")


def load_transport(optimization):
    namespace = {"__name__": "radial_transport_under_test", "__file__": str(SOURCE)}
    exec(compile(SOURCE.read_text(), str(SOURCE), "exec", optimize=optimization), namespace)
    return namespace


class TransportGuardTests(unittest.TestCase):
    def invoke_launch(self, root, optimization, quota, notebook=None):
        namespace = load_transport(optimization)
        transport = Mock(side_effect=lambda endpoint, payload=None: (
            quota if endpoint == "quota" else {"mock_launch": True}
        ))
        namespace["api"] = transport
        notebook_path = root / "fixture.ipynb"
        notebook_path.write_text(json.dumps(notebook if notebook is not None else {
            "nbformat": 4,
            "metadata": {"kernelspec": {"name": "python3"}},
            "cells": [],
        }))
        argv = [
            str(SOURCE), "launch", "ryangomez010/ripii-guard-fixture",
            str(root / "out"), "--notebook", str(notebook_path),
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            try:
                namespace["main"]()
            except ValueError as error:
                return transport, error
        return transport, None

    def test_paid_or_unknown_scaling_never_pushes_even_with_optimization(self):
        for optimization in (0, 2):
            for quota in (
                {"gpuQuota": {"isPayToScaleEnabled": True}},
                {"gpuQuota": {"isPayToScaleEnabled": None}},
                {"gpuQuota": {"isPayToScaleEnabled": 0}},
                {"gpuQuota": {}},
                {},
            ):
                with (
                    self.subTest(optimization=optimization, quota=quota),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    transport, error = self.invoke_launch(root, optimization, quota)
                    self.assertIsInstance(error, ValueError)
                    self.assertIn("Free GPU quota not confirmed", str(error))
                    self.assertEqual([call.args[0] for call in transport.call_args_list], ["quota"])
                    self.assertFalse((root / "out/launch.json").exists())

    def test_confirmed_free_quota_reaches_mock_push(self):
        for optimization in (0, 2):
            with (
                self.subTest(optimization=optimization),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                transport, error = self.invoke_launch(
                    root, optimization, {"gpuQuota": {"isPayToScaleEnabled": False}}
                )
                self.assertIsNone(error)
                self.assertEqual([call.args[0] for call in transport.call_args_list], ["quota", "push"])
                request = transport.call_args_list[1].args[1]
                self.assertEqual(request["machineShape"], "NvidiaTeslaT4")
                self.assertTrue(request["isPrivate"])
                self.assertFalse(request["enableInternet"])
                self.assertTrue((root / "out/launch.json").exists())

    def test_invalid_notebook_never_calls_api_even_with_optimization(self):
        for optimization in (0, 2):
            for notebook in ({"nbformat": 3}, {"nbformat": 4, "metadata": {}}):
                with (
                    self.subTest(optimization=optimization, notebook=notebook),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    transport, error = self.invoke_launch(
                        Path(tmp), optimization,
                        {"gpuQuota": {"isPayToScaleEnabled": False}}, notebook,
                    )
                    self.assertIsInstance(error, ValueError)
                    transport.assert_not_called()

    def test_non_https_download_never_opens_url_even_with_optimization(self):
        for optimization in (0, 2):
            with self.subTest(optimization=optimization):
                namespace = load_transport(optimization)
                namespace["api"] = Mock(side_effect=[
                    {"status": "complete"},
                    {"files": [{"fileName": "fixture-results.zip", "url": "http://example.invalid/archive"}]},
                ])
                with tempfile.TemporaryDirectory() as tmp:
                    argv = [str(SOURCE), "download", "ryangomez010/ripii-guard-fixture", tmp]
                    with (
                        patch.object(sys, "argv", argv),
                        patch("urllib.request.urlopen") as download,
                        contextlib.redirect_stdout(io.StringIO()),
                        self.assertRaisesRegex(ValueError, "HTTPS"),
                    ):
                        namespace["main"]()
                    download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
