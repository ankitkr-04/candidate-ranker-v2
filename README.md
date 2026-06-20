# candidate-ranker

Ranks the top *N* candidates (default 100) for a single job from a candidate pool and
writes a `candidate_id,rank,score,reasoning` CSV. The job's scoring policy is a
deterministic engine; the ~33 boolean judgments per candidate that need to read free text
come from a small language model. The work is split so the expensive part runs ahead of
time and the scored run is fast and reproducible on CPU.

## Two stages

```
precompute (GPU + network, no time limit)  ->  artifacts/<pool>/features.parquet
ranking    (CPU, no network, <=5 min, <=16 GB RAM)  ->  submission.csv
```

- **Precompute** parses the pool, normalizes fields, computes every deterministic feature
  and integrity signal, and runs the SLM for the boolean facts — one typed row per
  candidate, written to Parquet.
- **Ranking** scans that Parquet, compiles the entire policy into vectorized Polars
  expressions, sorts with the policy tie-break, and emits the top-N with grounded
  reasoning. No JSON parsing, no model, no network.

How it all fits together: [docs/architecture.md](docs/architecture.md).

## Install

Python 3.12.

```
# Ranking stage (CPU-only) and shared dependencies
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The precompute stage needs a GPU and a CUDA-matched vLLM build — see
[docs/precompute.md](docs/precompute.md).

## Run

1. Validate the job policy + integrity layer into artifacts (once; re-run after editing
   either source):

   ```
   python -m src.jd_parser.parse
   ```

2. Precompute features for a pool (`sample` / `1k` / `100k`):

   ```
   ./precompute.sh --pool 100k --dtype half          # GPU box
   ./precompute.sh --pool sample --no-slm             # CPU-only: deterministic features
   ```

3. Rank and write the submission:

   ```
   ./ranker.sh --pool 100k --out artifacts/100k/submission.csv
   ```

`./ranker.sh` wraps `python -m src.ranking.main`; useful flags: `--candidates <file>`,
`--top N`, `--debug` (writes the full scored ranking to `debug.jsonl`). Pools resolve to
`assets/candidates/<pool>_pool.{jsonl,json}`.

A fresh CPU-only environment can run end to end with no GPU: `precompute --no-slm` then
`ranker` produces a valid deterministic-only ranking (SLM flags defaulted by the policy).

## Layout

```
assets/           inputs: candidate pools, job policy, integrity penalties, schema
src/
  jd_parser/      validate JD + integrity source -> artifacts/tuning/*
  features/       deterministic feature derivation, integrity signals, utilities
  precompute/     parse + features + SLM -> features.parquet
  ranking/        compile policy -> score -> top-N CSV + reasoning
artifacts/        generated (gitignored): tuning/*, <pool>/features.parquet, submission.csv
```

## Docs

- [docs/architecture.md](docs/architecture.md) — scoring policy, vectorized compilation,
  the SLM pre-filter ceiling, tuning workflow.
- [docs/precompute.md](docs/precompute.md) — GPU setup, the SLM stage, incremental/resumable
  runs, flags, and the one-shot evidence repair utility.
- [docs/integrity.md](docs/integrity.md) — the job-agnostic plausibility penalty layer.
