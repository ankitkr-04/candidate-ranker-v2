"""Run the SLM over candidates with vLLM and guided JSON decoding.

GPU-only: vLLM is imported lazily so the rest of the pipeline (deterministic
features, ranker) runs in the CPU environment without it. Guided decoding against
the question schema makes every output valid, parseable JSON.

SlmRunner loads the model once so a caller can stream candidates through it in
batches (used by the precompute entrypoint to checkpoint long runs). The exact
structured-output API has shifted across vLLM releases; this targets the
GuidedDecodingParams interface (vLLM >= 0.6). Adjust the import if the installed
version renames it.
"""

import json

from src.models.candidate import Candidate
from src.models.policy import SlmQuestions
from src.precompute.download_model import ensure_model
from src.precompute.slm_input import build_messages, build_schema, slm_columns

_SUBJECT = "subject_of_primary_work"
_EVIDENCE = "evidence"


def _empty_fact(candidate: Candidate, flag_columns: list[str]) -> dict:
    fact: dict[str, object] = {"candidate_id": candidate.candidate_id, _SUBJECT: "", _EVIDENCE: ""}
    for flag in flag_columns:
        fact[flag] = False
    return fact


def _parse_output(candidate: Candidate, text: str, flag_columns: list[str]) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Guided decoding should prevent this; fall back to a null-equivalent fact.
        return _empty_fact(candidate, flag_columns)
    fact: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        _SUBJECT: data.get(_SUBJECT, ""),
        _EVIDENCE: data.get(_EVIDENCE, ""),
    }
    for flag in flag_columns:
        fact[flag] = bool(data.get(flag, False))
    return fact


class SlmRunner:
    """Loads Qwen3-4B once and answers the question set for batches of candidates."""

    def __init__(
        self,
        questions: SlmQuestions,
        tuning,
        *,
        model_dir=None,
        max_model_len: int = 8192,
        max_tokens: int = 640,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        from vllm.sampling_params import GuidedDecodingParams  # type: ignore[import-not-found]

        model_path = model_dir or ensure_model()
        self._llm = LLM(
            model=str(model_path),
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            guided_decoding=GuidedDecodingParams(json=build_schema(questions)),
        )
        self._questions = questions
        self._flag_columns = [c for c in slm_columns(tuning) if c not in (_SUBJECT, _EVIDENCE)]

    def generate(self, candidates: list[Candidate]) -> list[dict]:
        if not candidates:
            return []
        conversations = [build_messages(c, self._questions) for c in candidates]
        outputs = self._llm.chat(conversations, self._sampling)
        return [
            _parse_output(candidate, output.outputs[0].text if output.outputs else "", self._flag_columns)
            for candidate, output in zip(candidates, outputs)
        ]


def run_slm(candidates: list[Candidate], questions: SlmQuestions, tuning, **kwargs) -> list[dict]:
    """One-shot convenience wrapper: load the model and score all candidates."""
    if not candidates:
        return []
    return SlmRunner(questions, tuning, **kwargs).generate(candidates)
