"""Ranking entrypoint: read the feature parquet, score, and write the submission.

Reads the precomputed features, applies the vectorized scoring policy, sorts with
the policy's tie-break, and writes the top-N rows as
candidate_id,rank,score,reasoning. No candidate is dropped from scoring -- the
top-N is simply the head of the full ranking, so penalised and gated profiles
sink rather than disappear.

Examples:
  python -m src.ranking.main --pool sample
  python -m src.ranking.main --candidates assets/candidates/100k_pool.jsonl --top 100 --debug
"""

import argparse
from pathlib import Path

import polars as pl

from src.models.integrity import IntegrityPolicy, load_integrity
from src.models.tuning import Tuning
from src.paths import TUNING_ARTIFACT_DIR, pool_artifact_dir
from src.ranking.reasoning import compose_reasoning
from src.ranking.scorer import SCORE, score_frame

SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def load_tuning(path: Path | None = None) -> Tuning:
    tuning_path = path or (TUNING_ARTIFACT_DIR / "tuning.json")
    if not tuning_path.is_file():
        raise FileNotFoundError(
            f"{tuning_path} not found. Run `python -m src.jd_parser.parse` first."
        )
    return Tuning.model_validate_json(tuning_path.read_text())


def resolve_features(args: argparse.Namespace) -> tuple[str, Path]:
    """Return (pool_name, parquet_path) from --features / --pool / --candidates."""
    if args.features:
        path = Path(args.features)
        return path.parent.name, path
    if args.pool:
        return args.pool, pool_artifact_dir(args.pool) / "features.parquet"
    if args.candidates:
        name = Path(args.candidates).stem.removesuffix("_pool")
        return name, pool_artifact_dir(name) / "features.parquet"
    raise ValueError("Provide --features, --pool, or --candidates.")


def rank(
    frame: pl.DataFrame, tuning: Tuning, integrity: IntegrityPolicy | None, top_n: int
) -> pl.DataFrame:
    """Score and order the full frame; return it ranked (every candidate kept)."""
    scored = score_frame(frame, tuning, integrity)
    # Primary order is score; candidate_id asc breaks ties deterministically.
    ranked = scored.sort(
        [SCORE, "candidate_id"],
        descending=[True, False],
    ).with_row_index("rank", offset=1)
    return ranked


def build_submission(ranked: pl.DataFrame, top_n: int) -> pl.DataFrame:
    top = ranked.head(min(top_n, ranked.height))
    reasonings = [compose_reasoning(row) for row in top.iter_rows(named=True)]
    return top.with_columns(
        pl.col(SCORE).round(6).alias("score"),
        pl.Series("reasoning", reasonings),
    ).select(SUBMISSION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates and write the submission CSV.")
    parser.add_argument("--candidates", help="Candidates file (used to locate the pool's features).")
    parser.add_argument("--features", help="Path to features.parquet (overrides pool resolution).")
    parser.add_argument("--pool", help="Pool name, e.g. sample / 1k / 100k.")
    parser.add_argument("--tuning", help="Path to tuning.json (default artifacts/tuning/tuning.json).")
    parser.add_argument("--out", help="Output CSV path (default artifacts/<pool>/submission.csv).")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to output.")
    parser.add_argument("--debug", action="store_true", help="Also write the full scored ranking to debug.jsonl.")
    args = parser.parse_args()

    tuning = load_tuning(Path(args.tuning) if args.tuning else None)
    integrity = load_integrity()
    pool, features_path = resolve_features(args)
    if not features_path.is_file():
        raise FileNotFoundError(
            f"{features_path} not found. Run precompute first, e.g. "
            f"`python -m src.precompute.main --pool {pool} --no-slm`."
        )

    frame = pl.read_parquet(features_path)
    ranked = rank(frame, tuning, integrity, args.top)
    submission = build_submission(ranked, args.top)

    out_path = Path(args.out) if args.out else pool_artifact_dir(pool) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(out_path)
    print(f"Wrote {submission.height} ranked candidates to {out_path}")

    if args.debug:
        debug_path = pool_artifact_dir(pool) / "debug.jsonl"
        ranked.write_ndjson(debug_path)
        print(f"Wrote full scored ranking ({ranked.height} rows) to {debug_path}")


if __name__ == "__main__":
    main()
