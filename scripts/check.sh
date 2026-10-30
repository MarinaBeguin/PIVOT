#!/usr/bin/env bash
# Runs every quality gate. Usage: scripts/check.sh [python-bin-dir]
set -euo pipefail
BIN="${1:-$(dirname "$(command -v python)")}"
cd "$(dirname "$0")/.."
"$BIN/python" -m pytest
"$BIN/mypy"
"$BIN/pylint" src/pivot
# Test helpers are self-describing and imported as top-level modules by pytest.
"$BIN/pylint" tests --disable=missing-function-docstring,missing-param-doc,missing-return-doc,missing-yield-doc,\
wrong-import-order,too-many-locals
