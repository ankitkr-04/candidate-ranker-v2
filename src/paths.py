"""Filesystem locations shared across the pipeline.

Both stages resolve paths through this module so that precompute output and
ranking input always agree on where artifacts live.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = PROJECT_ROOT / "assets"
CANDIDATES_DIR = ASSETS_DIR / "candidates"
JOB_DIR = ASSETS_DIR / "job"
SCHEMA_DIR = ASSETS_DIR / "schema"
TUNING_DIR = ASSETS_DIR / "tuning"
INTEGRITY_SOURCE = ASSETS_DIR / "integrity" / "penalties.json"
MODEL_DIR = ASSETS_DIR / "model" / "Qwen3-4B-Instruct-2507"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TUNING_ARTIFACT_DIR = ARTIFACTS_DIR / "tuning"
INTEGRITY_ARTIFACT = TUNING_ARTIFACT_DIR / "integrity.json"

# Ranker output lives outside artifacts/: the parquet + tuning under artifacts/ are the
# expensive, committed inputs (no GPU recompute for collaborators), while rankings are a
# cheap, regenerable, git-ignored product written here.
RESULTS_DIR = PROJECT_ROOT / "results"


def pool_artifact_dir(pool: str) -> Path:
    """Return the per-pool artifact directory, e.g. artifacts/100k."""
    return ARTIFACTS_DIR / pool


def pool_result_dir(pool: str) -> Path:
    """Return the per-pool results directory, e.g. results/100k."""
    return RESULTS_DIR / pool
