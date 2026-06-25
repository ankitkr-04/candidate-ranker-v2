"""Prompt construction and fact merging for the SLM stage.

The model reads only the candidate's career history (per the policy's input_scope)
and answers a fixed question set. A JSON schema drives guided decoding so the
output is structurally valid and ordered: subject_of_primary_work and a one-line
evidence span first, then the boolean answers (per the policy's extraction_order).

Nothing here imports vLLM, so the merge helpers are usable in any environment.
"""

import polars as pl

from src.models.candidate import Candidate
from src.models.features import parquet_schema, slm_flag_columns
from src.models.policy import SlmQuestions
from src.models.tuning import Tuning

_SUBJECT = "subject_of_primary_work"
_EVIDENCE = "evidence"


def slm_columns(tuning: Tuning) -> list[str]:
    """SLM-derived columns: the text outputs followed by the boolean flags."""
    return [_SUBJECT, _EVIDENCE] + slm_flag_columns(tuning)


def career_history_text(candidate: Candidate) -> str:
    lines = []
    for index, role in enumerate(candidate.career_history, start=1):
        marker = " (current)" if role.is_current else ""
        lines.append(f"{index}. {role.title} at {role.company}{marker} - {role.duration_months} months")
        if role.description:
            lines.append(f"   {role.description}")
    return "\n".join(lines) if lines else "(no career history provided)"


def build_schema(questions: SlmQuestions) -> dict:
    """JSON schema for guided decoding; property order fixes the emission order."""
    properties: dict[str, dict] = {
        _SUBJECT: {"type": "string", "maxLength": 160},
        # Roomy enough to hold a complete sentence so the quote is not cut mid-word
        # in the reasoning display (reasoning.py also trims defensively).
        _EVIDENCE: {"type": "string", "maxLength": 320},
    }
    for question in questions.ask:
        if question.id in (_SUBJECT, _EVIDENCE):
            continue
        properties[question.id] = {"type": "boolean"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def build_messages(candidate: Candidate, questions: SlmQuestions) -> list[dict]:
    boolean_questions = [
        f"- {q.id}: {q.q}" for q in questions.ask if q.id not in (_SUBJECT, _EVIDENCE)
    ]
    subject_q = next((q.q for q in questions.ask if q.id == _SUBJECT), "")
    # Instructions and the question set are identical for every candidate, so they go in the
    # system message: vLLM prefix-caching computes this large block once and reuses it across
    # the pool, leaving only the short career history to prefill per candidate.
    system = (
        "You screen candidates for a senior AI/ML engineering role. Judge strictly from the "
        "candidate's career history. Do not assume facts that are not stated; when the history "
        "does not support a claim, answer false.\n\n"
        f"First, {_SUBJECT}: {subject_q}\n"
        "Then give an evidence span: quote one complete sentence from the history that you "
        "relied on. Quote it verbatim and do not stop mid-sentence.\n"
        "Then answer each question true or false:\n" + "\n".join(boolean_questions)
    )
    user = f"Career history:\n{career_history_text(candidate)}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _facts_schema(tuning: Tuning) -> dict:
    full = parquet_schema(tuning)
    columns = ["candidate_id"] + slm_columns(tuning)
    return {column: full[column] for column in columns}


def existing_slm_facts(parquet_path, tuning: Tuning) -> dict[str, dict]:
    """Map candidate_id -> SLM columns for rows already scored, for incremental runs.

    Robust to schema growth: when the question set gains a column the existing parquet
    does not have yet, only the columns actually present are read (the new ones are left
    unfilled). This lets a deterministic rebuild preserve prior facts, and lets a later
    SLM run backfill just the missing columns, instead of crashing on a missing column.
    """
    if not parquet_path.is_file():
        return {}
    available = set(pl.scan_parquet(parquet_path).collect_schema().names())
    columns = [c for c in (["candidate_id"] + slm_columns(tuning)) if c in available]
    if _SUBJECT not in columns:
        return {}
    done = pl.read_parquet(parquet_path, columns=columns).filter(
        pl.col(_SUBJECT).is_not_null()
    )
    return {row["candidate_id"]: row for row in done.iter_rows(named=True)}


def apply_slm_facts(table: pl.DataFrame, facts: list[dict], tuning: Tuning) -> pl.DataFrame:
    """Overwrite the SLM columns of table with the provided facts, matched by id."""
    if not facts:
        return table
    schema = _facts_schema(tuning)
    rows = [{column: fact.get(column) for column in schema} for fact in facts]
    facts_df = pl.DataFrame(rows, schema=schema)
    return table.update(facts_df, on="candidate_id")
