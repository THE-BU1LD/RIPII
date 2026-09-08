#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -s paper/MANUSCRIPT.md
echo 'manuscript source verified; no PDF/TeX submission artifact is claimed'
