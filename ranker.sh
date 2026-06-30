#!/usr/bin/env bash
# Run the ranking stage (CPU only, no network). All flags pass through to the ranker.
# Examples:
#   ./ranker.sh --pool 100k                       # -> results/100k/submission.csv
#   ./ranker.sh --pool 100k --format xlsx         # -> results/100k/submission.xlsx
#   ./ranker.sh --candidates assets/candidates/sample_pool.json --debug
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
exec "$PYTHON" -m src.ranking.main "$@"
