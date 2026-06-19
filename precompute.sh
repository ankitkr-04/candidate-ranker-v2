#!/usr/bin/env bash
# Run the precompute stage (parse, normalize, deterministic features, and SLM facts).
# GPU and network are permitted here. On the GPU box set PYTHON=.venv-gpu/bin/python.
# Examples:
#   ./precompute.sh --pool 100k
#   ./precompute.sh --pool sample --no-slm
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
exec "$PYTHON" -m src.precompute.main "$@"
