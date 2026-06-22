"""Config-driven profile-integrity features.

Computes the deterministic signals the integrity penalty layer scores against: date
consistency, tenure overruns, education-pair overlaps, skill durations that exceed the
candidate's experience or the tool's own age, and senior titles dated before the first
degree finished. All thresholds (overrun slack, seniority cutoff, tool era years) come
from the parsed IntegrityPolicy, so nothing here is hardcoded.

The deriver is built once per pool from an IntegrityPolicy and produces one dict per
candidate whose keys match the policy's declared flags and metrics.
"""

from datetime import date

from src.features.derive import _seniority_rank
from src.features.normalize import normalize_token
from src.models.candidate import Candidate
from src.models.integrity import IntegrityPolicy


class IntegrityDeriver:
    def __init__(self, integrity: IntegrityPolicy) -> None:
        self._slack = integrity.params.overrun_slack_months
        self._min_senior_rank = integrity.params.seniority_min_rank
        self._tool_eras = {normalize_token(name): year for name, year in integrity.tool_eras.items()}
        self._high_proficiency = {"expert", "advanced"}
        self._flags = list(integrity.features.flags)
        self._metrics = list(integrity.features.metrics)

    @property
    def columns(self) -> list[str]:
        return self._flags + self._metrics

    # Date-consistency signals ----------------------------------------------

    def _overrun_threshold(self, candidate: Candidate) -> float:
        return candidate.profile.years_of_experience * 12.0 + self._slack

    def end_before_start(self, candidate: Candidate) -> bool:
        return any(
            r.start_date and r.end_date and r.end_date < r.start_date
            for r in candidate.career_history
        )

    def career_months_overrun(self, candidate: Candidate) -> bool:
        total = sum(r.duration_months for r in candidate.career_history)
        return total > self._overrun_threshold(candidate)

    def role_months_overrun(self, candidate: Candidate) -> bool:
        durations = [r.duration_months for r in candidate.career_history]
        return (max(durations) if durations else 0) > self._overrun_threshold(candidate)

    def current_role_date_conflict(self, candidate: Candidate) -> bool:
        return any(
            (not r.is_current and r.end_date is None)
            or (r.is_current and r.end_date is not None)
            for r in candidate.career_history
        )

    # Education / seniority / skill signals ---------------------------------

    def _first_graduation_year(self, candidate: Candidate) -> int | None:
        years = [e.end_year for e in candidate.education if e.end_year is not None]
        return min(years) if years else None

    def senior_title_pre_graduation(self, candidate: Candidate) -> bool:
        # A senior title held before the earliest degree finished is implausible. Baselining
        # on the first degree (not the latest) avoids penalising a later part-time degree
        # taken while already senior.
        grad_year = self._first_graduation_year(candidate)
        if grad_year is None:
            return False
        return any(
            r.start_date is not None
            and r.start_date.year < grad_year
            and _seniority_rank(r.title) >= self._min_senior_rank
            for r in candidate.career_history
        )

    def num_education_overlaps(self, candidate: Candidate) -> float:
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

    def num_skill_anomalies(self, candidate: Candidate) -> float:
        # Skills claiming more months of use than the candidate's total experience.
        months_experience = candidate.profile.years_of_experience * 12.0
        return float(
            sum(
                1
                for s in candidate.skills
                if s.duration_months and s.duration_months > months_experience
            )
        )

    def num_proficiency_anomalies(self, candidate: Candidate) -> float:
        # A high proficiency claim with zero recorded use is implausible: you cannot be
        # expert/advanced at a tool you have used for 0 months. Generic data-quality check
        # (legitimate skills always carry a duration), not a honeypot special-case.
        return float(
            sum(
                1
                for s in candidate.skills
                if normalize_token(s.proficiency or "") in self._high_proficiency
                and not s.duration_months
            )
        )

    def num_skill_anachronisms(self, candidate: Candidate, reference_year: int) -> float:
        # Skills whose implied first-use year precedes the year the tool plausibly existed.
        # Only skills named in tool_eras are checked; the rest are ignored.
        count = 0
        for s in candidate.skills:
            if not s.duration_months:
                continue
            era = self._tool_eras.get(normalize_token(s.name))
            if era is None:
                continue
            implied_start = reference_year - s.duration_months / 12.0
            if implied_start < era:
                count += 1
        return float(count)

    # Assembly ---------------------------------------------------------------

    def compute(self, candidate: Candidate, reference: date) -> dict[str, object]:
        values: dict[str, object] = {
            "end_before_start": self.end_before_start(candidate),
            "career_months_overrun": self.career_months_overrun(candidate),
            "role_months_overrun": self.role_months_overrun(candidate),
            "current_role_date_conflict": self.current_role_date_conflict(candidate),
            "senior_title_pre_graduation": self.senior_title_pre_graduation(candidate),
            "num_education_overlaps": self.num_education_overlaps(candidate),
            "num_skill_anomalies": self.num_skill_anomalies(candidate),
            "num_proficiency_anomalies": self.num_proficiency_anomalies(candidate),
            "num_skill_anachronisms": self.num_skill_anachronisms(candidate, reference.year),
        }
        # Return only the columns the policy declares, so the asset stays the source of truth.
        return {column: values[column] for column in self.columns}
