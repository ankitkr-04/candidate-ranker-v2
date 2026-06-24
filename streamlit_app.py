"""Sandbox demo for the candidate ranker (Redrob hackathon, team CoreUse).

Entry point for Streamlit Cloud (auto-detects `streamlit_app.py` at the repo root).
It runs the ONLINE ranking stage only: load a pool's precomputed feature parquet,
apply the deterministic scoring policy, and produce a ranked CSV. The offline GPU
precompute (Qwen3-SLM) that builds the parquets is NOT part of this <=5-min CPU
budget -- see the note in the sidebar.

Local run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.integrity import load_integrity  # noqa: E402
from src.paths import (  # noqa: E402
    CANDIDATES_DIR,
    INTEGRITY_ARTIFACT,
    pool_artifact_dir,
    pool_result_dir,
)
from src.ranking.main import build_submission, load_tuning, rank  # noqa: E402

# Display order + metadata. `download` False => raw file too large to host.
POOL_META = {
    "sample": {"label": "sample · 50 curated", "kind": "curated", "download": True},
    "1k": {"label": "1k · 1,000 curated", "kind": "curated", "download": True},
    "100_rand_1": {"label": "100_rand_1 · 100 random", "kind": "random", "download": True},
    "100_rand_2": {"label": "100_rand_2 · 100 random", "kind": "random", "download": True},
    "100_rand_3": {"label": "100_rand_3 · 100 random", "kind": "random", "download": True},
    "1k_rand_1": {"label": "1k_rand_1 · 1,000 random", "kind": "random", "download": True},
    "1k_rand_2": {"label": "1k_rand_2 · 1,000 random", "kind": "random", "download": True},
    "1k_rand_3": {"label": "1k_rand_3 · 1,000 random", "kind": "random", "download": True},
    "100k": {"label": "100k · full pool (ranking only)", "kind": "full", "download": False},
}

JSONL_NAME = {
    "sample": "sample_pool.json",
    "1k": "1k_pool.jsonl",
    "100k": "100k_pool.jsonl",
}


def jsonl_path(pool: str) -> Path:
    return CANDIDATES_DIR / JSONL_NAME.get(pool, f"{pool}.jsonl")


def available_pools() -> list[str]:
    """Pools that have a precomputed feature parquet, in display order."""
    return [p for p in POOL_META if (pool_artifact_dir(p) / "features.parquet").is_file()]


@st.cache_resource(show_spinner=False)
def policy():
    return load_tuning(), load_integrity()


@st.cache_data(show_spinner=False)
def pool_size(pool: str) -> int:
    return pl.scan_parquet(pool_artifact_dir(pool) / "features.parquet").select(pl.len()).collect().item()


def run_ranking(pool: str, top_n: int):
    """Run the ranking stage, returning (submission_df, metrics dict)."""
    import os

    import psutil

    tuning, integrity = policy()
    proc = psutil.Process(os.getpid())
    parquet = pool_artifact_dir(pool) / "features.parquet"

    progress = st.progress(0, text="Loading precomputed features…")
    t0 = time.perf_counter()
    frame = pl.read_parquet(parquet)
    frame_mb = frame.estimated_size("mb")

    progress.progress(40, text=f"Scoring {frame.height:,} candidates…")
    ranked = rank(frame, tuning, integrity, top_n)

    progress.progress(80, text="Building submission…")
    submission = build_submission(ranked, top_n)
    elapsed = time.perf_counter() - t0

    out = pool_result_dir(pool) / "submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(out)
    progress.progress(100, text="Done.")
    progress.empty()

    metrics = {
        "elapsed_s": elapsed,
        "rows": frame.height,
        "frame_mb": frame_mb,
        "rss_mb": proc.memory_info().rss / 1024 / 1024,
        "out": out,
    }
    return submission, metrics


# ---------------------------------------------------------------------------- UI
st.set_page_config(page_title="Candidate Ranker · Sandbox", page_icon="🎯", layout="wide")

with st.sidebar:
    st.header("About this sandbox")
    st.markdown(
        "Runs the **online ranking stage** of a two-stage system: a precomputed "
        "feature parquet → deterministic scoring → ranked CSV. CPU-only, no network."
    )
    st.info(
        "The offline GPU precompute (Qwen3-SLM that builds the parquets) is **not** "
        "part of this ≤5-min CPU budget. The parquet is the committed handoff between "
        "the two stages.",
        icon="ℹ️",
    )
    if INTEGRITY_ARTIFACT.is_file():
        import json

        pen = json.loads(INTEGRITY_ARTIFACT.read_text()).get("penalties", [])
        st.caption(f"Integrity layer: {len(pen)} compounding penalties")
        st.caption("Score = base × Π(JD multipliers) × Π(integrity penalties). "
                   "Sort: score desc, candidate_id asc (deterministic).")

st.title("🎯 Candidate Ranker — Sandbox")
st.caption("Redrob hackathon · team CoreUse · Senior AI Engineer (Founding Team)")

pools = available_pools()
if not pools:
    st.error("No precomputed pools found under artifacts/. Run scripts/make_demo_pools.py.")
    st.stop()

col_a, col_b, col_c = st.columns([3, 2, 1])
with col_a:
    pool = st.selectbox(
        "Candidate pool",
        pools,
        format_func=lambda p: POOL_META[p]["label"],
    )
with col_b:
    n = pool_size(pool)
    top_n = st.number_input(
        "Top-N to output", min_value=1, max_value=n, value=min(100, n), step=10,
        help="The ranker keeps every candidate scored; Top-N is just the head of the full ranking.",
    )
with col_c:
    st.metric("Pool size", f"{n:,}")

run = st.button("▶ Run ranking", type="primary", width="stretch")

# Candidate-file view / download
raw = jsonl_path(pool)
with st.expander("Input candidate file", expanded=False):
    if not POOL_META[pool]["download"] or not raw.is_file():
        st.warning(
            "The raw 100k candidate file (~465 MB) is too large to host on the free "
            "sandbox tier, so it isn't downloadable here. Ranking still runs from the "
            "committed 2.5 MB feature parquet. The full file lives in the GitHub repo "
            "for the Stage-3 reproduction.",
            icon="⚠️",
        )
    else:
        data = raw.read_bytes()
        st.download_button(
            f"⬇ Download {raw.name} ({len(data) / 1024:.0f} KB)",
            data, file_name=raw.name, mime="application/json",
        )
        first = raw.read_text().splitlines()[:1]
        if first:
            st.code(first[0][:1200] + (" …" if len(first[0]) > 1200 else ""), language="json")
            st.caption("First record (truncated).")

if run:
    submission, m = run_ranking(pool, int(top_n))
    c1, c2, c3 = st.columns(3)
    c1.metric("Execution time", f"{m['elapsed_s']:.2f} s",
              help="Wall-clock for load → score → write. Budget: ≤5 min CPU.")
    c2.metric("Candidates scored", f"{m['rows']:,}")
    c3.metric("Process memory", f"{m['rss_mb']:.0f} MB",
              help=f"Feature frame in memory ≈ {m['frame_mb']:.1f} MB.")

    st.success(f"Ranked {m['rows']:,} candidates; wrote top {submission.height} to {m['out']}.")
    st.dataframe(submission.to_pandas(), width="stretch", hide_index=True)
    st.download_button(
        "⬇ Download submission.csv",
        submission.write_csv(), file_name=f"{pool}_submission.csv", mime="text/csv",
        type="primary",
    )
