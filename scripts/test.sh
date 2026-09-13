#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .test-tmp/system
export TMPDIR="$PWD/.test-tmp/system"
"${PYTHON:-.venv/bin/python}" -m pytest -q
