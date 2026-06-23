# candidate-ranker

Ranks the top N candidates (default 100) from a pool for a single ML engineering role and
writes `candidate_id,rank,score,reasoning` as a CSV. Designed for the Redrob Hackathon v4:
100k candidates, one job description, reproducible results on CPU in under 5 minutes.

---

## How it works

Two stages with a Parquet handoff:

```
precompute (GPU + network)     →     artifacts/<pool>/features.parquet
ranking    (CPU · ≤5 min · ≤16 GB · no network)     →     submission.csv
```

**Precompute** parses the candidate pool, normalizes noisy fields, computes every
deterministic feature and integrity signal, and runs a small language model (Qwen3-4B) to
get ~33 boolean judgments per candidate. Everything lands in one flat Parquet file — one
row per candidate.

**Ranking** scans that Parquet, compiles the entire scoring policy into vectorized Polars
expressions, scores all 100k candidates in a single pass, and emits the top-N rows with
grounded reasoning text. No JSON, no model, no network at this stage.

→ Full details: [docs/architecture.md](docs/architecture.md)

---

## Quick start

### 1 — Install (CPU ranking environment)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For GPU precompute, see [docs/precompute.md](docs/precompute.md).

### 2 — Parse the policy

```bash
python -m src.jd_parser.parse
```

Validates `assets/job/jd_parsed.json` and `assets/integrity/penalties.json` and writes
three artifacts to `artifacts/tuning/`. Run once; re-run whenever you edit either source.

### 3 — Precompute features

```bash
# CPU-only (deterministic features, no SLM) — works in .venv
./precompute.sh --pool sample --no-slm

# Full run with SLM on the GPU box
PYTHON=.venv-gpu/bin/python ./precompute.sh --pool 100k --dtype auto
```

### 4 — Rank

```bash
./ranker.sh --pool 100k
./ranker.sh --pool 100k --out results/100k/submission.csv --debug
```

### 5 — Validate submission

```bash
python -m src.features.validate_submission results/100k/submission.csv
```

---

## Repository layout

```
assets/
  job/jd_parsed.json          scoring policy (job-specific)
  integrity/penalties.json    plausibility penalties (job-agnostic)
  candidates/                 100k_pool.jsonl · 1k_pool.jsonl · sample_pool.json
  schema/candidate_schema.json
src/
  jd_parser/parse.py          validate policy sources → write artifacts/tuning/*
  models/                     Pydantic models: Candidate · Policy · Tuning · features
  features/                   normalize · metrics · derive · integrity · build · utilities
  precompute/                 parse pool → deterministic features → SLM → parquet
  ranking/                    compile policy → score → top-N CSV + reasoning
artifacts/                    generated
  tuning/                     tuning.json · slm_questions.json · integrity.json
  <pool>/                     features.parquet
results/                      generated (gitignored)
  <pool>/                     submission.csv 
precompute.sh                 wraps python -m src.precompute.main
ranker.sh                     wraps python -m src.ranking.main
requirements.txt              CPU ranking deps (pydantic · polars · orjson)
requirements-gpu.txt          GPU precompute deps (vllm · transformers · hf_hub)
```

---

## Documentation

| doc | covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | end-to-end data flow · scoring formula · multiplier types · predicate language · SLM pre-filter ceiling · Mermaid diagrams |
| [docs/precompute.md](docs/precompute.md) | GPU setup · vLLM + Qwen3-4B · SLM question schema · incremental/resumable runs · all flags · evidence repair |
| [docs/ranker.md](docs/ranker.md) | ranking step in detail · all CLI flags · scoring stages · reasoning composition · debug output |
| [docs/features.md](docs/features.md) | every file in src/features/ — normalize · metrics · derive · integrity · build · repair_evidence · export_csv · validate_submission |
| [docs/integrity.md](docs/integrity.md) | job-agnostic plausibility layer — design rationale · signals · penalty compounding · the prevalence/cliff test for adding a detector (with this dataset's honeypot-audit findings) · tuning |
