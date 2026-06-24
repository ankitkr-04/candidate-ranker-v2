"""Build small demo pools for the sandbox by sampling the SLM-scored subset.

Each demo pool is a random subset of the candidates that passed the precompute
ceiling pre-filter (non-empty `evidence`), so every candidate exercises the full
scoring path and produces a meaningful ranking -- no SLM re-run needed.

For each pool it writes:
  - artifacts/<pool>/features.parquet   (rows sliced from the 100k parquet)
  - assets/candidates/<pool>.jsonl      (raw records sliced from the 100k jsonl)

The candidate-id sets are reproducible (fixed seeds), so organisers can verify the
exact subset. Run from the repo root: `python scripts/make_demo_pools.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
POOL_JSONL = ROOT / "assets" / "candidates" / "100k_pool.jsonl"
POOL_PARQUET = ROOT / "artifacts" / "100k" / "features.parquet"
CANDIDATES_DIR = ROOT / "assets" / "candidates"
ARTIFACTS_DIR = ROOT / "artifacts"

# (pool_name, size, seed)
POOLS = [
    ("100_rand_1", 100, 101),
    ("100_rand_2", 100, 102),
    ("100_rand_3", 100, 103),
    ("1k_rand_1", 1000, 1001),
    ("1k_rand_2", 1000, 1002),
    ("1k_rand_3", 1000, 1003),
]


def main() -> None:
    feats = pl.read_parquet(POOL_PARQUET)
    scored = feats.filter(
        pl.col("evidence").is_not_null() & (pl.col("evidence").str.len_chars() > 0)
    )
    print(f"scored universe: {scored.height} candidates")

    # Sample id sets and write per-pool parquets.
    id_to_pools: dict[str, list[str]] = {}
    for name, size, seed in POOLS:
        sample = scored.sample(n=size, seed=seed)
        ids = sample["candidate_id"].to_list()
        (ARTIFACTS_DIR / name).mkdir(parents=True, exist_ok=True)
        sample.write_parquet(ARTIFACTS_DIR / name / "features.parquet")
        for cid in ids:
            id_to_pools.setdefault(cid, []).append(name)
        print(f"  {name}: {size} rows -> artifacts/{name}/features.parquet")

    # One streaming pass over the 465MB jsonl, fanning each needed record out to
    # every pool that wants it.
    writers = {name: (CANDIDATES_DIR / f"{name}.jsonl").open("w") for name, _, _ in POOLS}
    needed = len(id_to_pools)
    written = 0
    try:
        with POOL_JSONL.open() as fh:
            for line in fh:
                cid = json.loads(line)["candidate_id"]
                pools = id_to_pools.get(cid)
                if not pools:
                    continue
                for name in pools:
                    writers[name].write(line if line.endswith("\n") else line + "\n")
                written += 1
                if written == needed:
                    break
    finally:
        for w in writers.values():
            w.close()
    print(f"extracted raw records for {written}/{needed} unique candidates")

    # Verify each pool's jsonl row count matches its parquet.
    for name, size, _ in POOLS:
        n = sum(1 for _ in (CANDIDATES_DIR / f"{name}.jsonl").open())
        status = "ok" if n == size else f"MISMATCH (got {n})"
        print(f"  {name}.jsonl: {n} rows [{status}]")


if __name__ == "__main__":
    main()
