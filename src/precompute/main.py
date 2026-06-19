"""Precompute entrypoint: build the per-pool feature table.

Parses a candidate pool, computes the deterministic features, and writes
artifacts/<pool>/features.parquet. The SLM stage (Phase 3) fills the model-derived
flag columns; until then, or with --no-slm, those columns remain null and the
ranker treats them via the policy's uncertain handling.

Examples:
  python -m src.precompute.main --pool sample --no-slm
  python -m src.precompute.main --candidates assets/candidates/1k_pool.jsonl
"""

import argparse
from datetime import date
from pathlib import Path

import orjson
import polars as pl

from src.features.build import build_feature_row
from src.features.derive import FeatureDeriver
from src.models.candidate import Candidate
from src.models.features import parquet_schema
from src.models.tuning import Tuning
from src.paths import CANDIDATES_DIR, TUNING_ARTIFACT_DIR, pool_artifact_dir


def load_tuning() -> Tuning:
    path = TUNING_ARTIFACT_DIR / "tuning.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.jd_parser.parse` first."
        )
    return Tuning.model_validate_json(path.read_text())


def resolve_pool(pool: str | None, candidates: str | None) -> tuple[str, Path]:
    """Return (pool_name, candidates_path) from either --pool or --candidates."""
    if candidates:
        path = Path(candidates)
        name = path.stem.removesuffix("_pool")
        return name, path
    if not pool:
        raise ValueError("Provide either --pool or --candidates.")
    for suffix in (".jsonl", ".json"):
        path = CANDIDATES_DIR / f"{pool}_pool{suffix}"
        if path.is_file():
            return pool, path
    raise FileNotFoundError(f"No candidate file found for pool '{pool}' in {CANDIDATES_DIR}")


def load_candidates(path: Path, limit: int | None = None) -> list[Candidate]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    records.append(orjson.loads(line))
                if limit and len(records) >= limit:
                    break
    else:
        records = orjson.loads(path.read_bytes())
        if limit:
            records = records[:limit]
    return [Candidate.model_validate(r) for r in records]


def reference_date(candidates: list[Candidate]) -> date:
    """Most recent activity date in the pool; the recency baseline (0 days)."""
    dates = [c.redrob_signals.last_active_date for c in candidates if c.redrob_signals.last_active_date]
    return max(dates) if dates else date.today()


def build_feature_table(candidates: list[Candidate], tuning: Tuning) -> pl.DataFrame:
    deriver = FeatureDeriver(tuning)
    ref = reference_date(candidates)
    rows = [build_feature_row(c, deriver, ref) for c in candidates]
    return pl.DataFrame(rows, schema=parquet_schema(tuning))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the per-pool feature table.")
    parser.add_argument("--pool", help="Pool name, e.g. sample / 1k / 100k.")
    parser.add_argument("--candidates", help="Path to a candidates file (.jsonl or .json).")
    parser.add_argument("--out", help="Output parquet path (default artifacts/<pool>/features.parquet).")
    parser.add_argument("--limit", type=int, help="Process only the first N candidates (smoke test).")
    parser.add_argument("--no-slm", action="store_true", help="Skip the SLM stage (deterministic features only).")
    parser.add_argument("--force", action="store_true", help="Recompute SLM facts for all candidates.")
    args = parser.parse_args()

    tuning = load_tuning()
    pool, candidates_path = resolve_pool(args.pool, args.candidates)
    candidates = load_candidates(candidates_path, args.limit)

    table = build_feature_table(candidates, tuning)

    out_path = Path(args.out) if args.out else pool_artifact_dir(pool) / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_path)
    print(f"Wrote {table.height} rows x {table.width} cols to {out_path}")
    if not args.no_slm:
        print("Note: SLM stage not yet wired in; SLM flag columns are null. Use --no-slm to silence.")


if __name__ == "__main__":
    main()
