"""Deterministic metrics and honeypot flags derived from a candidate alone.

These depend only on the candidate's own data (dates, durations, signals,
education, skills) and not on any policy lookup, so they are kept separate from
the lookup-driven derivations in derive.py.
"""

from datetime import date
from statistics import median

from src.models.candidate import Candidate


def current_role_duration_months(candidate: Candidate) -> float:
    for role in candidate.career_history:
        if role.is_current:
            return float(role.duration_months)
    return 0.0


def median_tenure_last_3_months(candidate: Candidate) -> float:
    # career_history is stored most-recent-first, so the first three entries are
    # the last three employers.
    durations = [float(r.duration_months) for r in candidate.career_history[:3]]
    return float(median(durations)) if durations else 0.0


def last_active_days(candidate: Candidate, reference: date) -> float:
    last_active = candidate.redrob_signals.last_active_date
    if last_active is None:
        # NaN falls through every "<=" recency band to the default (worst) band.
        return float("nan")
    return float((reference - last_active).days)


def num_education_overlaps(candidate: Candidate) -> float:
    spans = [
        (e.start_year, e.end_year)
        for e in candidate.education
        if e.start_year is not None and e.end_year is not None
    ]
    overlaps = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            (a_start, a_end), (b_start, b_end) = spans[i], spans[j]
            if a_start <= b_end and b_start <= a_end:
                overlaps += 1
    return float(overlaps)


def num_skill_anomalies(candidate: Candidate) -> float:
    # Skills claiming more months of use than the candidate's total experience.
    # The policy's "older than the tool" check needs tool-release data the dataset
    # does not provide, so only the experience bound is applied here.
    months_experience = candidate.profile.years_of_experience * 12.0
    return float(
        sum(
            1
            for s in candidate.skills
            if s.duration_months and s.duration_months > months_experience
        )
    )


def honeypot_flags(candidate: Candidate) -> dict[str, bool]:
    durations = [r.duration_months for r in candidate.career_history]
    overrun_threshold = candidate.profile.years_of_experience * 12.0 + 18.0

    end_before_start = any(
        r.start_date and r.end_date and r.end_date < r.start_date
        for r in candidate.career_history
    )
    flag_conflict = any(
        (not r.is_current and r.end_date is None)
        or (r.is_current and r.end_date is not None)
        for r in candidate.career_history
    )
    return {
        "hp_end_before_start": end_before_start,
        "hp_career_overrun": sum(durations) > overrun_threshold,
        "hp_role_overrun": (max(durations) if durations else 0) > overrun_threshold,
        "hp_flag_conflict": flag_conflict,
    }
