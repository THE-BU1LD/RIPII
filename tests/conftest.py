from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# macOS commonly places pytest and subprocess temporary files on the smaller system
# data volume even when the repository and run artifacts live on external storage.
# Keep test-only temporary writes on the workspace volume for every pytest entry point.
_TEST_TMP = Path.cwd() / ".test-tmp" / "system"
_TEST_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TEST_TMP)
tempfile.tempdir = str(_TEST_TMP)

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
