"""Compose a short, grounded, *causal* reasoning string for a ranked candidate.

Runs only on the top-N rows, so per-row Python is affordable. Each entry is built
ONLY from values present in the scored feature row -- title, company, experience, the
SLM evidence span, which policy flags fired, and the multiplier/penalty stack the
scorer already emits (`mult__*`, `gate__*`) -- so it stays specific and never invents
facts (Stage-4: no hallucination).

This is the artifact a human reads at manual review of top submissions, so it is
organised around *cause and magnitude*, not as an exhaustive field dump:

    <verdict> — <one-line cause>. <title> at <company>, ~Xy exp (~Yy applied ML).
    <why this candidate is in the list (the base-score driver)>, but
      <why they are not ranked higher (the largest sub-1.0 multiplier), in recruiter
       terms with the concrete number behind it>.
    [Flags: <material data-quality / profile concerns only>.]
    Evidence: "<verbatim career-history span, trimmed at a word boundary>".

Two rules keep it honest and uncluttered:
  * it reads whichever stage dominates -- no hardcoded JD constants, no "if Meta then…";
  * multipliers within ~0.025 of 1.0 are immaterial and are suppressed, so a 0.99
    penalty is never framed as the reason a candidate ranks low.
The verdict is keyed to the final score (the value the ranking sorts on), so it stays
monotonic with rank.
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

# Profile-type concerns (job-fit, not data-quality). Surfaced in the Flags clause only
# when they fired and are not already the named dominant drag.
CONCERN_PHRASES = {
    "manager_not_builder": "reads as a manager rather than a hands-on builder",
    "research_not_applied": "research-only background with no production deployment",
    "primarily_adjacent": "was adjacent to the ML work rather than owning it",
    "observer_not_owner": "contributed without personally building the core work",
    "llm_api_wrapper_only": "AI experience is mostly LLM-API/wrapper work",
    "is_hobbyist_or_self_learner": "ML experience is self-taught, not professional",
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

# --- "why not ranked higher": label each JD multiplier / hard gate in recruiter terms.
# Static labels; stages with their own underlying numbers (availability, experience,
# notice, location) are filled in dynamically by _drag_phrase so the cause is concrete.
# Phrased as noun clauses so they read naturally after "downweighted ~X% on …" / "by …".
_DRAG_LABELS = {
    "current_title_congruence": "a current title that isn't ML-aligned",
    "company_context": "a weaker company-pedigree signal",
    "enterprise_lifer_overlay": "an enterprise-only career (startup-fit risk)",
    "work_mode_overlay": "a work-mode preference mismatch",
    "verification": "an unverified profile",
    "title_chaser": "frequent title/role changes",
    # hard gates
    "research_no_production": "a research background without production deployment",
    "lifelong_services": "a services/consulting (not product) background",
    "langchain_recent_only": "recent LLM-wrapper-only experience",
    "stale_coding_senior": "stale hands-on coding for the seniority",
}
# One- or two-word tag for the headline clause.
_SHORT_TAG = {
    "behavioral": "availability",
    "experience_band": "experience band",
    "applied_ml_years": "applied-ML depth",
    "location": "location",
    "notice_period": "notice period",
    "verification": "verification",
    "current_title_congruence": "title fit",
    "company_context": "company signal",
    "enterprise_lifer_overlay": "startup-fit risk",
    "work_mode_overlay": "work mode",
    "title_chaser": "role stability",
    "research_no_production": "no production work",
    "lifelong_services": "services background",
    "langchain_recent_only": "shallow LLM work",
    "stale_coding_senior": "stale coding",
}

# --- data-quality penalties (features/integrity.py): flag-valued, then count-valued.
_PENALTY_PHRASES = {
    "end_before_start_penalty": "a role ends before it starts",
    "career_months_overrun_penalty": "total tenure exceeds stated experience",
    "role_months_overrun_penalty": "a single role exceeds stated experience",
    "current_role_date_conflict_penalty": "inconsistent current-role dates",
    "experience_span_penalty": "stated experience exceeds the documented career",
    "seniority_before_graduation_penalty": "a senior title dated before graduation",
}
_PENALTY_COUNT_NOUNS = {
    "education_overlap_penalty": ("num_education_overlaps", "education-date overlap"),
    "skill_anomaly_penalty": ("num_skill_anomalies", "over-claimed skill"),
    "proficiency_anomaly_penalty": ("num_proficiency_anomalies", "unsupported expert-skill claim"),
    "skill_anachronism_penalty": ("num_skill_anachronisms", "anachronistic skill"),
}

# A multiplier this close to 1.0 moved the rank by <2.5% -- immaterial, so never framed
# as a cause. Penalties use the same bar so a 0.99 anachronism is suppressed.
_MATERIAL = 0.025

_SENTENCE_END = re.compile(r'[.!?]["\']?$')

# Strength selection: top candidates share the JD-core flags but differ in specialisms.
# Show ONE core anchor (so JD fit is explicit) then the candidate's more distinctive
# strengths first, so similar candidates surface different skills.
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
    if not phrases:
        if row.get("has_ai_native"):
            phrases.append("AI-native company background")
        elif row.get("has_product_company"):
            phrases.append("product-company background")
        elif row.get("current_title_bucket") in ("ideal", "strong_positive"):
            phrases.append("ML-aligned current title")
    return phrases


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _verdict(score: float) -> str:
    # Keyed to the final score so the verdict is always monotonic with rank.
    if score >= 0.5:
        return "Strong match"
    if score >= 0.4:
        return "Solid match"
    if score >= 0.3:
        return "Partial match"
    return "Limited match"


def _availability_phrase(row: dict) -> str:
    """The behavioral composite's concrete drivers, in recruiter terms."""
    bits: list[str] = []
    days = row.get("last_active_days")
    if days is not None and int(days) > 45:
        bits.append(f"{int(days)}-day inactivity")
    rate = row.get("recruiter_response_rate")
    if rate is not None and rate < 0.6:
        bits.append(f"{rate * 100:.0f}% recruiter-response")
    ic = row.get("interview_completion_rate")
    if ic is not None and ic < 0.5:
        bits.append(f"{ic * 100:.0f}% interview-completion")
    if row.get("open_to_work_flag") is False and not bits:
        bits.append("not openly on the market")
    if bits:
        return "weak availability (" + ", ".join(bits) + ")"
    return "weak engagement/availability signals"


def _drag_phrase(stage_id: str, row: dict) -> str:
    """Recruiter-language reason for a sub-1.0 stage, with its underlying number."""
    if stage_id == "behavioral":
        return _availability_phrase(row)
    if stage_id == "experience_band":
        yoe = row.get("years_of_experience") or 0.0
        return f"experience outside the ideal band (~{yoe:.0f}y)"
    if stage_id == "applied_ml_years":
        applied = row.get("applied_ml_years") or 0.0
        return f"limited applied-ML depth (~{applied:.1f}y)"
    if stage_id == "notice_period":
        return f"{int(row.get('notice_period_days') or 0)}-day notice period"
    if stage_id == "location":
        return _LOCATION_CONCERNS.get(
            row.get("location_relocation_bucket") or "", "location/relocation constraints"
        )
    return _DRAG_LABELS.get(stage_id, stage_id.replace("_", " "))


def _dominant_drag(row: dict) -> tuple[str, float] | None:
    """Largest fit/availability downweight among JD multipliers + hard gates.

    Integrity penalties are handled separately (the Flags clause) so the "why not
    higher" reason is about job fit and availability, not data-quality noise. Returns
    (stage_id, value) for the most material stage, or None if everything is ~1.0.
    """
    worst_id, worst_val = None, 1.0
    for col, val in row.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if col.startswith("gate__"):
            stage_id = col[len("gate__"):]
        elif col.startswith("mult__") and not col.endswith("_penalty"):
            stage_id = col[len("mult__"):]
        else:
            continue
        if val < worst_val:
            worst_id, worst_val = stage_id, float(val)
    if worst_id is not None and worst_val <= 1.0 - _MATERIAL:
        return worst_id, worst_val
    return None


def _material_flags(row: dict) -> list[str]:
    """Data-quality penalties that actually bit (suppress the immaterial ~0.99 ones)."""
    flags: list[str] = []
    for pen_id, phrase in _PENALTY_PHRASES.items():
        val = row.get(f"mult__{pen_id}")
        if isinstance(val, (int, float)) and val <= 1.0 - _MATERIAL:
            flags.append(phrase)
    for pen_id, (metric, noun) in _PENALTY_COUNT_NOUNS.items():
        val = row.get(f"mult__{pen_id}")
        count = int(row.get(metric) or 0)
        if isinstance(val, (int, float)) and val <= 1.0 - _MATERIAL and count > 0:
            flags.append(_plural(count, noun))
    return flags


def _clean_quote(text: str | None, limit: int = 260, source_cap: int = 190) -> str:
    """Return the SLM evidence span without a mid-word cut."""
    text = (text or "").strip()
    if not text:
        return ""
    over_limit = len(text) > limit
    if over_limit:
        text = text[:limit].rstrip()
    if not _SENTENCE_END.search(text):
        if (over_limit or len(text) >= source_cap) and " " in text:
            text = text[: text.rfind(" ")].rstrip(" ,;:-")
        text += "…"  # honest marker that this is an excerpt
    return text


def _base_clause(row: dict, base: float) -> str:
    """Why this candidate is in the list at all: the base-score driver, named."""
    strengths = _select_strengths(row)
    summary = "; ".join(strengths[:2])
    if base >= 0.97:
        return (f"{summary} (base fit maximal)" if summary
                else "top-tier substance (base fit maximal)")
    if base >= 0.85:
        return (f"{summary} (base near-maximal)" if summary
                else "strong substance (base near-maximal)")
    if base >= 0.6:
        return summary or "solid production ownership"
    return f"partial fit — {summary}" if summary else "limited production-ownership evidence"


def compose_reasoning(row: dict) -> str:
    title = (row.get("current_title") or "Candidate").strip()
    company = (row.get("current_company") or "").strip()
    yoe = row.get("years_of_experience") or 0.0
    applied = row.get("applied_ml_years") or 0.0
    score = row.get("score") or 0.0
    base = row.get("base_score")
    base = float(base) if base is not None else score

    lead = title + (f" at {company}" if company else "")
    lead += f", ~{yoe:.1f} yrs exp (~{applied:.1f} yrs applied ML)"

    drag = _dominant_drag(row)
    base_clause = _base_clause(row, base)
    flags = _material_flags(row)

    # Headline: the single cause framing a reviewer needs first. A drag only earns the
    # headline when it materially moved the rank (>=7%); a ~5% trim is body-only so it
    # never overstates the knock on an otherwise top candidate.
    strong_drag = drag is not None and (1.0 - drag[1]) >= 0.07
    if strong_drag:
        tag = _SHORT_TAG.get(drag[0], _drag_phrase(drag[0], row))
        headline = (f"strong substance, held back by {tag}" if base >= 0.85
                    else f"held back by {tag}")
    elif flags:
        headline = ("strong substance, data-quality flags noted" if base >= 0.85
                    else "data-quality flags noted")
    elif base < 0.6:
        headline = "limited ownership evidence"
    elif base >= 0.97:
        headline = "top-of-pool fit"
    else:
        headline = "clears the bar across the board"

    # Why-in-list, then why-not-higher in one causal sentence.
    cause = base_clause[0].upper() + base_clause[1:]
    if drag is not None:
        pct = round((1.0 - drag[1]) * 100)
        verb = (f"downweighted ~{pct}% on" if pct >= 5 else "reduced by")
        cause += f", but {verb} {_drag_phrase(drag[0], row)}"
    cause += "."

    parts = [f"{_verdict(score)} — {headline}. {lead}.", cause]

    # Flags: material data-quality penalties (computed above) first, then any profile-fit
    # concern not already named. Capped so the line stays scannable.
    for flag, phrase in CONCERN_PHRASES.items():
        if _is_true(row, flag) and phrase not in flags:
            flags.append(phrase)
    if flags:
        parts.append("Flags: " + ", ".join(flags[:3]) + ".")
    elif base >= 0.9 and drag is not None and drag[1] < 0.85:
        # High substance pulled low by fit/availability, nothing wrong with the data --
        # say so, so a reviewer knows the low rank isn't a red flag.
        parts.append("No material data-quality flags.")

    quote = _clean_quote(row.get("evidence"))
    if quote:
        parts.append(f'Evidence: "{quote}".')
    return " ".join(parts)
