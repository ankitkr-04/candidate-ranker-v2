"""Compose a short, grounded, *causal* reasoning string for a ranked candidate.

Runs only on the top-N rows, so per-row Python is affordable. Each entry is built
ONLY from values present in the scored feature row -- title, company, experience, which
policy flags fired, and the multiplier/penalty stack the scorer already emits
(`mult__*`, `gate__*`) -- so it stays specific and never invents facts (Stage-4: no
hallucination).

This is the artifact a human reads at manual review of top submissions, so it is
organised around *cause and magnitude*, not as an exhaustive field dump. Each entry
leads with the verdict (kept monotonic with rank), names why the candidate is in the
list (the base-score driver) and why they are not ranked higher (the largest sub-1.0
multiplier, in recruiter terms with the concrete number behind it), and lists only
material data-quality flags.

Three rules keep it honest and un-templated:
  * it reads whichever stage dominates -- no hardcoded JD constants, no "if Meta then…";
  * multipliers within ~0.025 of 1.0 are immaterial and suppressed, so a 0.99 penalty is
    never framed as the reason a candidate ranks low;
  * the sentence frame is chosen per candidate (decorrelated from rank), so sampled
    reasonings are structurally as well as factually different from one another.
"""

import re

# Positive drivers, keyed to the live SLM flag set. Base-tier (JD "absolutely need")
# and bonus-tier (JD "we'd like") flags both surface here; the tier split only governs
# scoring, not how a fired strength is described to a reviewer.
POSITIVE_PHRASES = {
    "owns_retrieval_prod": "owns production retrieval",
    "owns_ranking_prod": "owns a production ranking/recommendation system",
    "owns_eval_framework": "built ranking/retrieval evaluation",
    "vector_db_prod": "ran vector-DB infrastructure in production",
    "shipped_endtoend_at_scale": "shipped an end-to-end system at scale",
    "retrieval_ops_depth": "deep retrieval-operations experience",
    "strong_python_prod": "strong production Python engineering",
    "ab_testing_ml": "ran A/B tests on ML systems",
    "ltr_experience": "learning-to-rank experience",
    "reranker_twostage": "built a two-stage retrieve-then-rerank pipeline",
    "relevance_judgment_pipeline": "ran a relevance-judgment/labeling pipeline",
    "llm_finetuning": "hands-on LLM fine-tuning",
    "distributed_systems_scale": "distributed-systems-at-scale experience",
    "hrtech_or_marketplace_exp": "HR-tech/marketplace experience",
    "external_validation": "papers, talks, or open-source contributions",
}

# Profile-type concerns (job-fit, not data-quality). Surfaced in the flags clause only
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

# Location drag, by location_relocation_bucket. Phrased as noun clauses so they read
# naturally after "downweighted ~X% due to …".
_LOCATION_PHRASES = {
    "tier1_relocating": "a tier-1 location requiring relocation",
    "tier1_not_relocating": "relocation constraints (tier-1 city, not open to relocating)",
    "other_india_relocating": "an out-of-hub location requiring relocation",
    "other_india_not_relocating": "relocation constraints (outside the hub, not open to relocating)",
    "outside_relocating": "a location outside India",
    "outside_not_relocating": "a location outside India with no relocation",
}

# Title drag, by current_title_bucket. The role targets senior/staff/lead ML titles
# (the `ideal` bucket); `strong_positive` titles (ML/AI/Search/Recsys Engineer) are
# ML-aligned but one notch below that seniority -- so the phrasing is about seniority and
# exactness of match, never "not ML-aligned". `junior_ml` is core-ML by domain (e.g.
# "Junior ML Engineer") but junior for a senior founding role -- the drag is seniority,
# NOT a domain mismatch, so it must never read as "not explicitly ML".
_TITLE_PHRASES = {
    "strong_positive": "a current title one notch below the target seniority",
    "junior_ml": "an explicitly-ML title at junior seniority (below the senior bar)",
    "moderate_positive": "an ML-adjacent current title (data/analytics, not core ML engineering)",
    "neutral_read_description": "a generic current title (not explicitly ML)",
    "heavy_penalty": "a largely off-target current title",
    "hard_sink": "a current title unrelated to ML/AI",
}

# Static recruiter labels for the remaining JD multipliers / hard gates (noun clauses).
_DRAG_LABELS = {
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
    "current_title_congruence": "title seniority",
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
    "skill_anachronism_magnitude_penalty": ("num_skill_anachronisms", "anachronistic skill"),
}
# Anachronism is scored by two compounding stages -- time magnitude and how many skills --
# but surfaced under one noun. Gate the noun on the product so a count-only-material case
# (several marginal overruns) is still attributed, never silently dropped.
_PENALTY_COMPANION = {
    "skill_anachronism_magnitude_penalty": "skill_anachronism_count_penalty",
}

# A multiplier this close to 1.0 moved the rank by <2.5% -- immaterial, so never framed
# as a cause. Penalties use the same bar so a 0.99 anachronism is suppressed.
_MATERIAL = 0.025

# Strength selection: top candidates share the JD-core flags but differ in specialisms.
# Anchors are the base-tier "absolutely need" cores; specialists are the bonus-tier
# "we'd like" differentiators surfaced after the anchor so two cores read distinctly.
_ANCHOR_STRENGTHS = ("owns_retrieval_prod", "owns_ranking_prod", "owns_eval_framework", "vector_db_prod")
_SPECIALIST_STRENGTHS = (
    "reranker_twostage", "ltr_experience", "relevance_judgment_pipeline", "external_validation",
    "llm_finetuning", "distributed_systems_scale", "retrieval_ops_depth",
    "hrtech_or_marketplace_exp", "shipped_endtoend_at_scale",
)


def _is_true(row: dict, flag: str) -> bool:
    return row.get(flag) is True


def _cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


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
        elif row.get("current_title_bucket") in ("ideal", "strong_positive", "junior_ml"):
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
    """Name the behavioral composite's *material* drivers, in recruiter terms.

    The `behavioral` stage multiplies several signals (recency, recruiter response,
    interview completion, open-to-work, recent applications). The dominant-drag % is the
    NET composite effect, so naming inactivity alone is misleading: two candidates can
    share an identical recency factor yet differ in net drag because of the OTHER signals
    (e.g. one is open-to-work and applying, the other is neither). We therefore surface
    every signal that is actually pulling the composite down -- ordered worst-first, each
    gated at the knee where its curve starts biting -- so the stated causes match the
    magnitude and never imply a day-count difference that the bands treat as immaterial.
    """
    drivers: list[tuple[float, str]] = []  # (drop magnitude, phrase); larger = worse
    days = row.get("last_active_days")
    if days is not None and int(days) > 30:  # recency band knee (<=30d is clean)
        drop = 0.55 if int(days) > 180 else (0.30 if int(days) > 90 else 0.10)
        drivers.append((drop, f"{int(days)}-day inactivity"))
    rate = row.get("recruiter_response_rate")
    if rate is not None and rate < 0.6:
        drop = 0.55 if rate < 0.25 else (0.22 if rate < 0.45 else 0.08)
        drivers.append((drop, f"{rate * 100:.0f}% recruiter-response"))
    ic = row.get("interview_completion_rate")
    if ic is not None and ic < 0.9:  # completion curve bites below 0.9, not 0.5
        drop = 0.08 if ic < 0.5 else 0.04
        drivers.append((drop, f"{ic * 100:.0f}% interview-completion"))
    if row.get("open_to_work_flag") is False:  # a real 5% drag -- never hide it
        drivers.append((0.05, "not openly on the market"))
    apps = row.get("applications_submitted_30d")
    if apps is not None and int(apps) == 0:
        drivers.append((0.03, "no recent applications"))
    drivers.sort(key=lambda d: d[0], reverse=True)
    bits = [phrase for _, phrase in drivers[:3]]
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
        return f"a {int(row.get('notice_period_days') or 0)}-day notice period"
    if stage_id == "location":
        return _LOCATION_PHRASES.get(
            row.get("location_relocation_bucket") or "", "location/relocation constraints"
        )
    if stage_id == "current_title_congruence":
        return _TITLE_PHRASES.get(
            row.get("current_title_bucket") or "", "a current title that's an imperfect match"
        )
    return _DRAG_LABELS.get(stage_id, stage_id.replace("_", " "))


def _dominant_drag(row: dict) -> tuple[str, float] | None:
    """Largest fit/availability downweight among JD multipliers + hard gates.

    Integrity penalties are handled separately (the flags clause) so the "why not
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


def _all_flags(row: dict) -> list[str]:
    """Material data-quality penalties first, then any profile-fit concern that fired."""
    flags: list[str] = []
    for pen_id, phrase in _PENALTY_PHRASES.items():
        val = row.get(f"mult__{pen_id}")
        if isinstance(val, (int, float)) and val <= 1.0 - _MATERIAL:
            flags.append(phrase)
    for pen_id, (metric, noun) in _PENALTY_COUNT_NOUNS.items():
        val = row.get(f"mult__{pen_id}")
        companion = row.get(f"mult__{_PENALTY_COMPANION.get(pen_id, '')}")
        if isinstance(val, (int, float)) and isinstance(companion, (int, float)):
            val *= companion
        count = int(row.get(metric) or 0)
        if isinstance(val, (int, float)) and val <= 1.0 - _MATERIAL and count > 0:
            flags.append(_plural(count, noun))
    for flag, phrase in CONCERN_PHRASES.items():
        if _is_true(row, flag) and phrase not in flags:
            flags.append(phrase)
    return flags[:3]


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


def _headline(base: float, drag, strong_drag: bool, flags: list[str], row: dict) -> str:
    if strong_drag:
        tag = _SHORT_TAG.get(drag[0], _drag_phrase(drag[0], row))
        return (f"strong substance, held back by {tag}" if base >= 0.85
                else f"held back by {tag}")
    if flags:
        return ("strong substance, data-quality flags noted" if base >= 0.85
                else "data-quality flags noted")
    if base < 0.6:
        return "limited ownership evidence"
    if base >= 0.97:
        return "top-of-pool fit"
    return "clears the bar across the board"


def _drag_predicate(pct: int, phrase: str) -> str:
    """Body phrasing for the dominant drag (>=5% reads as an explicit downweight)."""
    return f"downweighted ~{pct}% due to {phrase}" if pct >= 5 else f"reduced by {phrase}"


def _frame_canonical(ctx: dict) -> str:
    """Verdict — headline. Title at Company, ~Xy exp (~Yy applied ML). Base, but drag."""
    lead = ctx["titlecompany"] + f", ~{ctx['yoe']:.1f} yrs exp (~{ctx['applied']:.1f} yrs applied ML)"
    cause = _cap(ctx["base_clause"])
    if ctx["drag"] is not None:
        cause += f", but {_drag_predicate(ctx['pct'], ctx['drag_phrase'])}"
    parts = [f"{ctx['verdict']} — {ctx['headline']}. {lead}.", cause + "."]
    if ctx["flags"]:
        parts.append("Flags: " + ", ".join(ctx["flags"]) + ".")
    elif ctx["reassure"]:
        parts.append("No material data-quality flags.")
    return " ".join(parts)


def _frame_profile_led(ctx: dict) -> str:
    """Verdict. Title at Company (~Yy applied ML…): base clause. Drag sentence."""
    exp = f"~{ctx['applied']:.1f} yrs applied ML"
    if ctx["yoe"] > ctx["applied"] + 0.3:
        exp += f" of ~{ctx['yoe']:.1f} total"
    parts = [f"{ctx['verdict']}. {ctx['titlecompany']} ({exp}): {ctx['base_clause']}."]
    if ctx["drag"] is not None:
        parts.append(_cap(_drag_predicate(ctx["pct"], ctx["drag_phrase"])) + ".")
    if ctx["flags"]:
        parts.append("Data-quality notes: " + ", ".join(ctx["flags"]) + ".")
    elif ctx["reassure"]:
        parts.append("No material data-quality flags.")
    return " ".join(parts)


_FRAMES = (_frame_canonical, _frame_profile_led)


def compose_reasoning(row: dict) -> str:
    title = (row.get("current_title") or "Candidate").strip()
    company = (row.get("current_company") or "").strip()
    score = row.get("score") or 0.0
    base = row.get("base_score")
    base = float(base) if base is not None else score

    drag = _dominant_drag(row)
    flags = _all_flags(row)
    strong_drag = drag is not None and (1.0 - drag[1]) >= 0.07

    ctx = {
        "verdict": _verdict(score),
        "titlecompany": title + (f" at {company}" if company else ""),
        "yoe": float(row.get("years_of_experience") or 0.0),
        "applied": float(row.get("applied_ml_years") or 0.0),
        "base": base,
        "base_clause": _base_clause(row, base),
        "headline": _headline(base, drag, strong_drag, flags, row),
        "drag": drag,
        "pct": round((1.0 - drag[1]) * 100) if drag else 0,
        "drag_phrase": _drag_phrase(drag[0], row) if drag else "",
        "flags": flags,
        # High substance pulled low purely by fit/availability, data clean -- say so, so a
        # reviewer knows the low rank isn't a red flag.
        "reassure": base >= 0.9 and drag is not None and drag[1] < 0.85 and not flags,
    }

    # Frame chosen from the candidate id, so structural variety is deterministic yet
    # decorrelated from rank (no visible rank%N rotation).
    digits = re.sub(r"\D", "", str(row.get("candidate_id") or "0")) or "0"
    return _FRAMES[int(digits) % len(_FRAMES)](ctx)
