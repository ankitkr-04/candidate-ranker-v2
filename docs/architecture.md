# Architecture

Two stages with a Parquet handoff. Precompute does all the heavy work ahead of time;
the ranker is fast, deterministic, and reproducible on CPU.

```
precompute (GPU + network, no time limit)        ranking (CPU, no network, <=5 min)
  parse JSONL  -> typed Candidate                  scan features.parquet
  normalize    (geo / company / title)             compile policy -> Polars expressions
  deterministic features (metrics, buckets)        score + sort + tie-break
  integrity signals (job-agnostic plausibility)    top-N rows -> reasoning
  SLM facts    (33 booleans + evidence span)       submission.csv
       \__________ features.parquet ___________________/
```

## Scoring policy

The job policy (`assets/job/jd_parsed.json`, parsed to `artifacts/tuning/tuning.json`) is a
deterministic engine:

```
base_score = clamp(career_substance + skill_booster, 0, 1)
score      = base_score
             x multiplier stages      (title, company, location, work-mode, ...)
             x hard gates             (e.g. lifelong-services)
             x integrity penalties    (job-agnostic; see docs/integrity.md)
             then honeypot-zeroed if any hp_* flag fires
```

- **`career_substance`** is the *only* SLM-dependent part of the score: its additive
  ownership flags and internal gates all come from the model. Its theoretical max is 1.0.
- **`skill_booster`** is gated on `career_substance >= 0.6`.
- Every multiplier stage and hard gate reads **deterministic** features only.

No candidate is ever removed. Honeypots and gates drive a score toward 0 but the row stays
in the ranking, so penalised profiles sink rather than disappear.

## Vectorized compilation

The whole policy compiles to Polars expressions over a flat, wide fact table (one column
per flag / metric / categorical). There is no per-row Python over the pool:

- `ranking/predicate.py` — `compile_predicate(pred) -> pl.Expr` (flag/metric leaves,
  `all`/`any`/`not`, comparison ops).
- `ranking/scorer.py` — each multiplier `type` (`lookup`, `curve`, `conditional`, `decay`,
  `composite_product`) compiles to a `pl.Expr`; gates and integrity penalties are extra
  stages; one lazy query produces `score` plus a per-stage breakdown column for debug.

100k candidates score in milliseconds. The only Python loop is reasoning over the <=100
selected rows (`ranking/reasoning.py`), composed from the breakdown columns + the SLM
evidence span so the text stays grounded.

## SLM pre-filter (the "ceiling")

Because every multiplier and gate is deterministic, a candidate's best *possible* score is
exact: `ceiling = base_score(=1.0) x all multiplier stages x integrity penalties`, with
gates assumed 1.0 (over-estimate, so it never drops a viable candidate). Precompute runs
the SLM only on candidates whose ceiling `>= --slm-ceiling` (default 0.02); the rest keep
null SLM columns, score ~0, and stay ranked. This avoids spending GPU time on candidates
that cannot reach the top N even with a perfect model result. See
[precompute.md](precompute.md).

## Tuning workflow

- Numeric knobs (weights, multiplier bands, gate multipliers, thresholds) — re-run the
  **ranker** only (seconds).
- Lookup membership / normalization — re-run the **CPU feature build**
  (`precompute --no-slm`, no GPU, preserves cached SLM facts).
- The SLM step is the only expensive part, and it is incremental/resumable.

## Missing SLM facts

The ranker defaults absent SLM flags via the policy's `uncertain_treatment` (positive ->
`false`, disqualifier -> does not fire), yielding a valid deterministic-only ranking. The
real scored run ships a complete Parquet, so it never relies on this; the path exists for a
fresh CPU-only sandbox (`precompute --no-slm` then rank).
