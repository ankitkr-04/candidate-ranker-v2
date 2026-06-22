"""Compose a short, grounded reasoning string for a ranked candidate.

Runs only on the top-N rows, so per-row Python is affordable. Each entry is built
ONLY from values present in the feature row -- title, company, experience, the
behavioural signals, the SLM evidence span, and which policy flags fired -- so it
stays specific and never invents facts (Stage-4: no hallucination). The shape is:

    <verdict>. <title> at <company>, ~Xy exp (~Yy applied ML).
    Strengths: <flags that fired, in JD terms>.
    Signals: <specific behavioural numbers -- the JD says to weigh these>.
    Concerns: <honest gaps, with counts where we have them>.
    Evidence: "<verbatim career-history span, trimmed at a word boundary>".

The verdict is keyed to the final score (the value the ranking sorts on) so it is always
monotonic with rank; the per-candidate signal numbers, evidence span, and the specific gaps
in Concerns give each entry substantive variation rather than a shared template.
"""

import re

POSITIVE_PHRASES = {
    "owns_retrieval_prod": "owns production retrieval",
    "owns_ranking_prod": "owns a production ranking/recommendation system",
    "owns_eval_framework": "built ranking/retrieval evaluation",
    "vector_db_prod": "ran vector-DB infrastructure in production",
    "shipped_endtoend_at_scale": "shipped an end-to-end system at scale",
    "retrieval_ops_depth": "deep retrieval-operations experience",
    "ltr_experience": "learning-to-rank experience",
    "reranker_twostage": "built a two-stage retrieve-then-rerank pipeline",
    "llm_finetuning": "hands-on LLM fine-tuning",
    "realtime_ml_serving": "real-time ML serving experience",
    "prod_ml_ops": "production MLOps experience",
    "hrtech_or_marketplace_exp": "HR-tech/marketplace experience",
    "external_validation": "papers, talks, or open-source contributions",
}

CONCERN_PHRASES = {
    "manager_not_builder": "reads as a manager rather than a hands-on builder",
    "research_not_applied": "research-only background with no production deployment",
    "primarily_adjacent": "was adjacent to the ML work rather than owning it",
    "observer_not_owner": "contributed without personally building the core work",
    "llm_api_wrapper_only": "AI experience is mostly LLM-API/wrapper work",
    "is_hobbyist_or_self_learner": "ML experience is self-taught, not professional",
    "enterprise_lifer": "entire career at large enterprises (Series-A fit risk)",
    "cv_dominant": "ML focus is computer vision, not retrieval/ranking",
    "speech_dominant": "ML focus is speech/audio, not retrieval/ranking",
    "robotics_dominant": "ML focus is robotics, not retrieval/ranking",
}

_LOCATION_CONCERNS = {
    "tier1_not_relocating": "in a tier-1 city but not open to relocating",
    "other_india_not_relocating": "elsewhere in India and not open to relocating",
    "outside_not_relocating": "outside India and not open to relocating",
    "outside_relocating": "based outside India",
}

# Job-agnostic data-quality penalties (features/integrity.py). Flag-valued first, then
# count-valued; phrased so the reasoning names the specific implausibility, not a label.
INTEGRITY_FLAG_CONCERNS = {
    "end_before_start": "a role's end date precedes its start date",
    "career_months_overrun": "total role tenure exceeds the stated experience",
    "role_months_overrun": "a single role exceeds the stated experience",
    "current_role_date_conflict": "inconsistent current-role dates",
    "senior_title_pre_graduation": "a senior title dated before the first degree finished",
}
# Count-valued integrity concerns: phrased as a noun so the actual count can lead
# (e.g. "2 anachronistic skills"), which adds both specificity and per-row variation.
INTEGRITY_COUNT_NOUNS = {
    "num_skill_anachronisms": "anachronistic skill",
    "num_education_overlaps": "education-date overlap",
    "num_skill_anomalies": "over-claimed skill",
}

_SENTENCE_END = re.compile(r'[.!?]["\']?$')


# Strength selection: top candidates share the JD-core flags but differ in specialisms.
# Show ONE core anchor (so JD fit is explicit) then the candidate's more distinctive
# strengths first, so similar candidates surface different skills instead of a shared triple.
_ANCHOR_STRENGTHS = ("owns_retrieval_prod", "owns_ranking_prod", "owns_eval_framework", "vector_db_prod")
_SPECIALIST_STRENGTHS = (
    "reranker_twostage", "realtime_ml_serving", "external_validation", "llm_finetuning",
    "retrieval_ops_depth", "ltr_experience", "hrtech_or_marketplace_exp", "prod_ml_ops",
    "shipped_endtoend_at_scale",
)


def _is_true(row: dict, flag: str) -> bool:
    return row.get(flag) is True


def _select_strengths(row: dict) -> list[str]:
    fired = [flag for flag in POSITIVE_PHRASES if _is_true(row, flag)]
    ordered: list[str] = []
    for flag in _ANCHOR_STRENGTHS:  # one JD-core anchor
        if flag in fired:
            ordered.append(flag)
            break
    ordered += [f for f in _SPECIALIST_STRENGTHS if f in fired and f not in ordered]
    ordered += [f for f in fired if f not in ordered]  # backfill remaining cores

    phrases = [POSITIVE_PHRASES[f] for f in ordered[:3]]
    if len(phrases) < 3:
        if row.get("has_ai_native"):
            phrases.append("AI-native company background")
        elif row.get("has_product_company"):
            phrases.append("product-company background")
    if not phrases and row.get("current_title_bucket") in ("ideal", "strong_positive"):
        phrases.append("ML-aligned current title")
    return phrases[:3]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _verdict(score: float) -> str:
    # Keyed to the final score -- the value the ranking sorts on -- so the verdict is
    # always monotonic with rank (rank 1 can never read below rank 100). The score is a
    # compressed product of many sub-1.0 factors, so the bands are calibrated to its
    # top-of-pool distribution; the logistic gaps that pulled it down are spelled out in
    # the Signals/Concerns clauses.
    if score >= 0.5:
        return "Strong match"
    if score >= 0.4:
        return "Solid match"
    if score >= 0.3:
        return "Partial match"
    return "Limited match"


def _engagement(row: dict) -> list[str]:
    """Specific behavioural-signal phrases. They vary per candidate (so they break the
    templated-reasoning problem) and the JD explicitly asks to weigh availability."""
    bits: list[str] = []
    days = row.get("last_active_days")
    if days is not None:
        days = int(days)
        bits.append(
            "active today" if days <= 1
            else f"active {days}d ago" if days <= 45
            else f"inactive {days}d"
        )
    rate = row.get("recruiter_response_rate")
    if rate is not None:
        bits.append(f"{rate * 100:.0f}% recruiter response")
    github = row.get("github_activity_score")
    if github is not None and github >= 0:
        bits.append(f"GitHub {github:.0f}")
    else:
        saves = row.get("saved_by_recruiters_30d") or 0
        if saves > 0:
            bits.append(f"{int(saves)} recruiter saves/30d")
    return bits[:3]


def _clean_quote(text: str | None, limit: int = 260, source_cap: int = 190) -> str:
    """Return the SLM evidence span without a mid-word cut. The model's length cap can
    end a span mid-token; trim to the last whole word and mark it with an ellipsis."""
    text = (text or "").strip()
    if not text:
        return ""
    over_limit = len(text) > limit
    if over_limit:
        text = text[:limit].rstrip()
    if not _SENTENCE_END.search(text):
        # A span that hit the model's length cap ends mid-token -> drop the partial word.
        if (over_limit or len(text) >= source_cap) and " " in text:
            text = text[: text.rfind(" ")].rstrip(" ,;:-")
        text += "…"  # honest marker that this is an excerpt, not a full sentence
    return text


def compose_reasoning(row: dict) -> str:
    title = (row.get("current_title") or "Candidate").strip()
    company = (row.get("current_company") or "").strip()
    yoe = row.get("years_of_experience") or 0.0
    applied = row.get("applied_ml_years") or 0.0
    score = row.get("score") or 0.0

    lead = title + (f" at {company}" if company else "")
    # Both to one decimal: applied_ml_years is capped at years_of_experience, so matching
    # the precision keeps the parenthetical from ever reading as larger than the total.
    lead += f", ~{yoe:.1f} yrs exp (~{applied:.1f} yrs applied ML)"

    strengths = _select_strengths(row)

    concerns = [phrase for flag, phrase in CONCERN_PHRASES.items() if _is_true(row, flag)]
    if row.get("current_title_bucket") in ("hard_sink", "heavy_penalty"):
        concerns.append("current title is not ML-focused")
    location_concern = _LOCATION_CONCERNS.get(row.get("location_relocation_bucket") or "")
    if location_concern:
        concerns.append(location_concern)
    notice = row.get("notice_period_days") or 0
    if notice >= 90:
        concerns.append(f"{int(notice)}-day notice period")
    for flag, phrase in INTEGRITY_FLAG_CONCERNS.items():
        if _is_true(row, flag):
            concerns.append(phrase)
    for metric, noun in INTEGRITY_COUNT_NOUNS.items():
        count = int(row.get(metric) or 0)
        if count > 0:
            concerns.append(_plural(count, noun))

    parts = [f"{_verdict(score)}. {lead}."]
    if strengths:
        parts.append("Strengths: " + ", ".join(strengths[:3]) + ".")
    engagement = _engagement(row)
    if engagement:
        parts.append("Signals: " + ", ".join(engagement) + ".")
    if concerns:
        parts.append("Concerns: " + ", ".join(concerns[:3]) + ".")
    quote = _clean_quote(row.get("evidence"))
    if quote:
        parts.append(f'Evidence: "{quote}".')
    return " ".join(parts)
