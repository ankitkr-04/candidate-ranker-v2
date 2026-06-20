"""
Export artifact parquet files to CSV for human inspection.

A standalone utility, NOT a precompute stage — run it on demand:

    # Export all parquet files from a pool
    python -m src.tools.export_csv --artifacts artifacts/full_pool

    # Export only specific files (anywhere under the pool)
    python -m src.tools.export_csv --artifacts artifacts/full_pool --files profiles skills

    # Export a file inside a subdirectory (e.g., ingest/career_history)
    python -m src.tools.export_csv --artifacts artifacts/full_pool --files ingest/career_history

CSV cannot hold nested types, so List columns (e.g. canonical_keys) are joined
into "a; b". CSV is for EYES ONLY — it is lossy (dtypes, nesting); the pipeline
always reads the parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def csv_safe(df: pl.DataFrame) -> pl.DataFrame:
    """Convert nested columns to JSON or delimited strings for CSV export."""
    casts = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.List):
            casts.append(pl.col(name).cast(pl.List(pl.Utf8)).list.join("; ").alias(name))
        elif isinstance(dtype, pl.Struct):
            casts.append(pl.col(name).struct.json_encode().alias(name))
    return df.with_columns(casts) if casts else df


def export(artifacts: Path, output: Path | None, files: list[str] | None) -> None:
    """
    Recursively find parquet files under `artifacts` and export them to CSV.

    Args:
        artifacts: root directory (e.g., artifacts/full_pool)
        output: output directory (default: artifacts/converted)
        files: optional list of filenames or relative paths to export (basename or subpath)
    """
    if output is None:
        output = artifacts / "converted"

    if files:
        # Resolve each file specification
        targets = []
        for f in files:
            # Try as relative path under artifacts
            p = artifacts / f
            if p.exists() and p.suffix == ".parquet":
                targets.append(p)
                continue
            # Try as basename with recursive glob
            if not f.endswith(".parquet"):
                f = f + ".parquet"
            found = list(artifacts.rglob(f))
            if not found:
                raise SystemExit(f"No parquet file matching: {f}")
            targets.extend(found)
    else:
        # Recursively find all parquet files
        targets = list(artifacts.rglob("*.parquet"))

    if not targets:
        raise SystemExit(f"No parquet files found under {artifacts}")

    output.mkdir(parents=True, exist_ok=True)

    for p in targets:
        # Use relative path to avoid collisions (replace '/' with '_')
        rel = p.relative_to(artifacts)
        safe_name = str(rel).replace("/", "_").replace(".parquet", ".csv")
        dest = output / safe_name

        df = csv_safe(pl.read_parquet(p))
        df.write_csv(dest)
        print(f"{rel!s:40s} -> {dest}  ({df.height} rows, {df.width} cols)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export artifact parquet files to CSV")
    ap.add_argument("--artifacts", type=Path, required=True,
                    help="Root directory containing parquet files (e.g., artifacts/full_pool)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory (default: <artifacts>/converted)")
    ap.add_argument("--files", nargs="+", default=None,
                    help="Specific parquet files to export (basename or relative path)")
    args = ap.parse_args()
    export(args.artifacts, args.output, args.files)


if __name__ == "__main__":
    main()