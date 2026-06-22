"""Compose a short, grounded reasoning string for a ranked candidate.

Runs only on the top-N rows, so per-row Python is affordable. Every clause is
built from values actually present in the feature row -- title, company,
experience, location, notice period, the SLM evidence span, and which policy
flags fired -- so the text stays specific and never invents facts. When SLM flags
are absent (deterministic-only ranking) the reasoning leans on the deterministic
signals.
"""

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
INTEGRITY_COUNT_CONCERNS = {
    "num_skill_anachronisms": "claims a skill predating the technology",
    "num_education_overlaps": "overlapping education dates",
    "num_skill_anomalies": "a skill claimed for longer than the stated experience",
}


def _is_true(row: dict, flag: str) -> bool:
    return row.get(flag) is True


def compose_reasoning(row: dict) -> str:
    title = (row.get("current_title") or "Candidate").strip()
    company = (row.get("current_company") or "").strip()
    yoe = row.get("years_of_experience") or 0.0
    applied = row.get("applied_ml_years") or 0.0

    lead = title
    if company:
        lead += f" at {company}"
    lead += f", ~{yoe:.0f} yrs experience (~{applied:.1f} yrs applied ML)"

    strengths = [phrase for flag, phrase in POSITIVE_PHRASES.items() if _is_true(row, flag)]
    if row.get("has_ai_native"):
        strengths.append("worked at an AI-native company")
    elif row.get("has_product_company"):
        strengths.append("product-company background")
    if not strengths and row.get("current_title_bucket") in ("ideal", "strong_positive"):
        strengths.append("ML-aligned current title")

    concerns = [phrase for flag, phrase in CONCERN_PHRASES.items() if _is_true(row, flag)]
    if row.get("current_title_bucket") in ("hard_sink", "heavy_penalty"):
        concerns.append("current title is not ML-focused")
    location_concern = _LOCATION_CONCERNS.get(row.get("location_relocation_bucket") or "")
    if location_concern:
        concerns.append(location_concern)
    notice = row.get("notice_period_days") or 0
    if notice >= 90:
        concerns.append(f"long notice period ({int(notice)} days)")
    for flag, phrase in INTEGRITY_FLAG_CONCERNS.items():
        if _is_true(row, flag):
            concerns.append(phrase)
    for metric, phrase in INTEGRITY_COUNT_CONCERNS.items():
        if (row.get(metric) or 0) > 0:
            concerns.append(phrase)

    parts = [lead + "."]
    if strengths:
        parts.append("Strengths: " + ", ".join(strengths[:3]) + ".")
    if concerns:
        parts.append("Concerns: " + ", ".join(concerns[:3]) + ".")
    evidence = (row.get("evidence") or "").strip()
    if evidence:
        parts.append(f'Evidence: "{evidence}".')
    return " ".join(parts)
