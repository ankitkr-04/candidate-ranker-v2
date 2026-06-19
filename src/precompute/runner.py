"""Run the SLM over candidates with vLLM and guided JSON decoding.

GPU-only: vLLM is imported lazily so the rest of the pipeline (deterministic
features, ranker) runs in the CPU environment without it. Guided decoding against
the question schema makes every output valid, parseable JSON.

The exact structured-output API has shifted across vLLM releases; this targets the
GuidedDecodingParams interface (vLLM >= 0.6). Adjust the import here if the
installed version renames it.
"""

import json

from src.models.candidate import Candidate
from src.models.policy import SlmQuestions
from src.precompute.download_model import ensure_model
from src.precompute.slm_input import build_messages, build_schema, slm_columns


def _empty_fact(candidate: Candidate, flag_columns: list[str]) -> dict:
    fact: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "subject_of_primary_work": "",
        "evidence": "",
    }
    for flag in flag_columns:
        fact[flag] = False
    return fact


def run_slm(
    candidates: list[Candidate],
    questions: SlmQuestions,
    tuning,
    *,
    model_dir=None,
    max_model_len: int = 8192,
    max_tokens: int = 640,
    gpu_memory_utilization: float = 0.90,
) -> list[dict]:
    """Return one fact dict per candidate (candidate_id + SLM columns)."""
    if not candidates:
        return []

    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    from vllm.sampling_params import GuidedDecodingParams  # type: ignore[import-not-found]

    model_path = model_dir or ensure_model()
    llm = LLM(
        model=str(model_path),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        guided_decoding=GuidedDecodingParams(json=build_schema(questions)),
    )

    conversations = [build_messages(c, questions) for c in candidates]
    outputs = llm.chat(conversations, sampling)

    flag_columns = [c for c in slm_columns(tuning) if c not in ("subject_of_primary_work", "evidence")]
    facts: list[dict] = []
    for candidate, output in zip(candidates, outputs):
        text = output.outputs[0].text if output.outputs else ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Guided decoding should prevent this; fall back to a null-equivalent fact.
            facts.append(_empty_fact(candidate, flag_columns))
            continue
        fact: dict[str, object] = {"candidate_id": candidate.candidate_id}
        fact["subject_of_primary_work"] = data.get("subject_of_primary_work", "")
        fact["evidence"] = data.get("evidence", "")
        for flag in flag_columns:
            fact[flag] = bool(data.get(flag, False))
        facts.append(fact)
    return facts
