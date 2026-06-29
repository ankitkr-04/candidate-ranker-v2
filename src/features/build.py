"""Assemble one flat feature row per candidate.

build_feature_row produces a dict whose keys match models.features.parquet_schema:
deterministic flags and metrics are computed here, while the SLM flags and text
outputs are left null and filled later by the precompute SLM stage.
"""

from datetime import date

from src.features.derive import FeatureDeriver
from src.features.integrity import IntegrityDeriver
from src.features.metrics import (
    current_role_duration_months,
    last_active_days,
    median_tenure_last_3_months,
)
from src.models.candidate import Candidate


def build_feature_row(
    candidate: Candidate,
    deriver: FeatureDeriver,
    reference_date: date,
    integrity_deriver: IntegrityDeriver | None = None,
) -> dict:
    signals = candidate.redrob_signals
    row: dict[str, object] = {"candidate_id": candidate.candidate_id}

    row.update(
        {
            "current_is_services": deriver.current_is_services(candidate),
            "majority_career_services": deriver.majority_career_services(candidate),
            "has_product_company": deriver.has_product_company(candidate),
            "has_ai_native": deriver.has_ai_native(candidate),
            "titles_escalating": deriver.titles_escalating(candidate),
            "is_local": deriver.is_local(candidate),
            "prefers_remote": signals.preferred_work_mode == "remote",
            "open_to_work_flag": signals.open_to_work_flag,
            "enterprise_lifer": deriver.enterprise_lifer(candidate),
        }
    )

    # Job-agnostic plausibility signals (date consistency, education/skill anomalies).
    if integrity_deriver is not None:
        row.update(integrity_deriver.compute(candidate, reference_date))

    # SLM flags are unknown until the model runs; left null for uncertain handling.
    for flag in deriver.slm_flag_names:
        row[flag] = None

    row.update(
        {
            "years_of_experience": candidate.profile.years_of_experience,
            "applied_ml_years": deriver.applied_ml_years(candidate),
            "median_tenure_last_3_months": median_tenure_last_3_months(candidate),
            "current_role_duration_months": current_role_duration_months(candidate),
            "last_active_days": last_active_days(candidate, reference_date),
            "recruiter_response_rate": signals.recruiter_response_rate,
            "interview_completion_rate": signals.interview_completion_rate,
            "saved_by_recruiters_30d": float(signals.saved_by_recruiters_30d),
            "applications_submitted_30d": float(signals.applications_submitted_30d),
            "notice_period_days": float(signals.notice_period_days),
            "github_activity_score": signals.github_activity_score,
            "num_qualifying_unevidenced_skills": deriver.num_qualifying_unevidenced_skills(candidate),
            "priority_assessment_signal": deriver.priority_assessment_signal(candidate),
        }
    )

    row.update(
        {
            "current_title_bucket": deriver.current_title_bucket(candidate),
            "location_relocation_bucket": deriver.location_relocation_bucket(candidate),
            "verification_state": deriver.verification_state(candidate),
        }
    )

    row.update(
        {
            "current_title": candidate.profile.current_title,
            "current_company": candidate.profile.current_company,
            "location": candidate.profile.location,
            "country": candidate.profile.country,
            "preferred_work_mode": signals.preferred_work_mode,
            "willing_to_relocate": signals.willing_to_relocate,
        }
    )

    row["subject_of_primary_work"] = None
    row["evidence"] = None
    return row
