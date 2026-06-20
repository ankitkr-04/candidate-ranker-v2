# Precompute (GPU stage)

Builds `artifacts/<pool>/features.parquet`: one typed row per candidate with deterministic
features, integrity signals, and the SLM facts. GPU and network are permitted here; this
runs ahead of the time-boxed ranking step.

## Environment

Python 3.12, on the GPU box. vLLM must match the GPU driver's CUDA version, so install it
with `uv` (which inspects the driver) rather than plain pip:

```
python -m venv .venv-gpu
.venv-gpu/bin/pip install uv
.venv-gpu/bin/uv pip install vllm --torch-backend=auto
.venv-gpu/bin/pip install -r requirements.txt transformers "huggingface_hub[hf_transfer]"
```

The model (`Qwen/Qwen3-4B-Instruct-2507`) downloads on first use into
`assets/model/Qwen3-4B-Instruct-2507/` (idempotent).

Set `PYTHON=.venv-gpu/bin/python` so `precompute.sh` uses the GPU environment.

## Running

```
# full run for a pool (sample / 1k / 100k)
PYTHON=.venv-gpu/bin/python ./precompute.sh --pool 100k --dtype half

# deterministic features only (no GPU) -- also the CPU-sandbox path
./precompute.sh --pool sample --no-slm
```

`precompute.sh` wraps `python -m src.precompute.main`.

| flag | default | purpose |
|---|---|---|
| `--pool` | — | pool name; resolves `assets/candidates/<pool>_pool.{jsonl,json}` |
| `--candidates` | — | explicit candidates file (instead of `--pool`) |
| `--out` | `artifacts/<pool>/features.parquet` | output Parquet |
| `--limit N` | — | process only the first N candidates (smoke test) |
| `--no-slm` | off | deterministic features only; preserves cached SLM facts |
| `--force` | off | recompute SLM facts for all candidates |
| `--batch-size` | 1000 | SLM candidates per checkpoint |
| `--slm-ceiling` | 0.02 | run the SLM only on candidates whose ceiling is at least this |
| `--dtype` | auto | vLLM dtype; use `half` on GPUs without bf16 (e.g. Tesla T4) |
| `--max-model-len` | 4096 | vLLM max sequence length; lower raises concurrency |
| `--max-tokens` | 512 | max new tokens generated per candidate |

## SLM stage

The model reads **only** the candidate's career history (per the policy's `input_scope`)
and answers a fixed question set. Guided JSON decoding fixes the output shape and order:
`subject_of_primary_work` and a one-line `evidence` span first, then the boolean answers.
The instructions are identical per candidate, so vLLM prefix-caching computes that block
once for the whole pool.

**Incremental / resumable.** Facts already in the Parquet are skipped; only
`selected - cached` is computed, and a checkpoint is written every `--batch-size`
candidates. Cancelling loses only the in-flight batch. `--force` wipes the cache and
recomputes everything. Lowering `--slm-ceiling` later only computes the *additional*
candidates — you never pay twice.

**The ceiling pre-filter** (`--slm-ceiling`) is an exact upper bound on the achievable
score (see [architecture.md](architecture.md)); candidates below it cannot reach the top N
even with a perfect model result, so the SLM is skipped for them. They keep null SLM
columns and remain in the ranking at score ~0.

## Cleaning SLM evidence (one-shot, no re-run)

Qwen3 occasionally code-switches into Chinese inside the free-text `evidence` span. That
field is display-only — the scorer never reads it — so it is fixed as a pure data edit, not
a model re-run:

```
python -m src.features.repair_evidence --parquet artifacts/100k/features.parquet
#   --dry-run    preview the transforms and counts, write nothing
#   --no-backup  skip the features.parquet.bak copy
```

It backs up the Parquet, translates the frequent phrases to English (the `_RESTORE` map),
truncates rare mid-word fragments at the leak, and writes back **only** the `evidence`
column (all other columns byte-identical, SLM facts and scores untouched). If a new token
shows up in a future run, add one line to `_RESTORE` and re-run.

## Inspection utilities

```
python -m src.features.export_csv --artifacts artifacts/100k        # Parquet -> CSV (eyes only)
python -m src.features.validate_submission artifacts/100k/submission.csv
```
