"""Deterministic, job-specific metrics derived from a candidate alone.

These depend only on the candidate's own data (dates, durations, signals) and not on
any policy lookup, so they are kept separate from the lookup-driven derivations in
derive.py. Job-agnostic plausibility signals (date consistency, education/skill
anomalies) live in features/integrity.py instead.
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
