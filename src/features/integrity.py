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


def _to_fractional_year(value: int | str) -> float:
    """Normalize a tool-era / company-founding entry to a fractional year.

    Accepts a bare year (int or "YYYY") or a "YYYY-MM" / "YYYY-MM-DD" string. A year with no
    month is read as that January (month 1) -- the earliest the tool/company plausibly existed,
    which keeps the anachronism/predates checks lenient when only the year is known. A stated
    month maps to the fraction of the year it begins (Jan -> .0, Dec -> .917), so a real release
    or founding month tightens the boundary without ever fabricating precision the data lacks.
    """
    if isinstance(value, int):
        return float(value)
    parts = value.split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    return year + (month - 1) / 12.0


def _date_to_fractional_year(d: date) -> float:
    """Candidate-side dates carry a real month, so reduce them to the same fractional-year scale
    as the era maps -- preserving the month instead of truncating to the calendar year."""
    return d.year + (d.month - 1) / 12.0


class IntegrityDeriver:
    def __init__(self, integrity: IntegrityPolicy) -> None:
        self._slack = integrity.params.overrun_slack_months
        self._min_senior_rank = integrity.params.seniority_min_rank
        self._span_buffer = integrity.params.experience_span_buffer_years
        self._anachronism_buffer_years = integrity.params.anachronism_buffer_months / 12.0
        self._anachronism_grace_years = integrity.params.anachronism_grace_years
        self._anomaly_buffer_months = integrity.params.anomaly_buffer_months
        self._company_predates_buffer_years = (
            integrity.params.company_predates_buffer_months / 12.0
        )
        self._tool_eras = {
            normalize_token(name): _to_fractional_year(era)
            for name, era in integrity.tool_eras.items()
        }
        self._company_founding = {
            normalize_token(name): _to_fractional_year(founded)
            for name, founded in integrity.company_founding.items()
        }
        self._high_proficiency = {"expert", "advanced"}
        self._seniority_ladder = integrity.seniority_ladder
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

    def experience_exceeds_career_span(self, candidate: Candidate, reference: date) -> bool:
        # The stated experience cannot exceed the span from the earliest documented role to
        # the reference date by more than the buffer: those years have to come from somewhere.
        # A normal candidate who drops an early job shows a gap of a year or two; a claim that
        # overshoots the whole documented career by 5+ years is not a memory lapse, it is invented.
        starts = [r.start_date for r in candidate.career_history if r.start_date is not None]
        if not starts:
            return False
        span_years = (reference - min(starts)).days / 365.25
        return candidate.profile.years_of_experience > span_years + self._span_buffer

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
            and _seniority_rank(r.title, self._seniority_ladder) >= self._min_senior_rank
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
        # Skills claiming more months of use than the candidate's total experience, beyond a
        # buffer. A tool used somewhat longer than the first paid job is ordinary (college,
        # side-projects, hackathons, open-source between jobs), so the buffer absorbs that noise
        # and the count registers only genuine, large over-claims.
        threshold = (
            candidate.profile.years_of_experience * 12.0 + self._anomaly_buffer_months
        )
        return float(
            sum(
                1
                for s in candidate.skills
                if s.duration_months and s.duration_months > threshold
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

    def _skill_overruns(self, candidate: Candidate, reference: float) -> list[float]:
        # Per skill named in tool_eras, the number of years its claimed tenure predates the tool's
        # own era, net of the per-skill grace buffer. Only beyond-buffer (positive) overruns are
        # returned: a skill within the buffer of its era is not anachronistic. Skills not in
        # tool_eras, and those without a stated duration, are ignored. `reference` is the fractional
        # year of the run's reference date, so the implied start keeps month precision on both sides.
        overruns: list[float] = []
        for s in candidate.skills:
            if not s.duration_months:
                continue
            era = self._tool_eras.get(normalize_token(s.name))
            if era is None:
                continue
            implied_start = reference - s.duration_months / 12.0
            overrun = era - implied_start - self._anachronism_buffer_years
            if overrun > 0:
                overruns.append(overrun)
        return overruns

    def num_skill_anachronisms(self, candidate: Candidate, reference: float) -> float:
        # How many skills are claimed for longer than the tool has existed (beyond the buffer).
        return float(len(self._skill_overruns(candidate, reference)))

    def skill_anachronism_years(self, candidate: Candidate, reference: float) -> float:
        # Total beyond-buffer overrun in years -- the magnitude the time penalty scales against,
        # so being a year ahead is gentler than ten. The aggregate grace forgives a *single*
        # anachronism only: one tool known early via closed-source/pre-release exposure is
        # plausible, several are not, so with two or more the full sum is charged with no relief.
        overruns = self._skill_overruns(candidate, reference)
        if not overruns:
            return 0.0
        if len(overruns) == 1:
            return max(0.0, overruns[0] - self._anachronism_grace_years)
        return float(sum(overruns))

    def years_predating_company(self, candidate: Candidate) -> float:
        # Largest number of years, net of the per-role buffer, by which a role's start predates its
        # company's founding. Internal-consistency checks cannot catch this (the dates are
        # self-consistent); only the external founding date reveals that the company did not exist
        # yet. Both sides are taken to month precision -- the role's actual start month against the
        # founding date's fractional year -- so a start a few months ahead of the public founding
        # date is absorbed by the buffer rather than rounded up to a full year. Companies not in the
        # map are skipped, so this never fires on the pool's fictional placeholders.
        worst = 0.0
        for r in candidate.career_history:
            if r.start_date is None or not r.company:
                continue
            founded = self._company_founding.get(normalize_token(r.company))
            if founded is None:
                continue
            gap = founded - _date_to_fractional_year(r.start_date) - self._company_predates_buffer_years
            if gap > worst:
                worst = gap
        return float(worst)

    # Assembly ---------------------------------------------------------------

    def compute(self, candidate: Candidate, reference: date) -> dict[str, object]:
        reference_frac = _date_to_fractional_year(reference)
        values: dict[str, object] = {
            "end_before_start": self.end_before_start(candidate),
            "career_months_overrun": self.career_months_overrun(candidate),
            "role_months_overrun": self.role_months_overrun(candidate),
            "current_role_date_conflict": self.current_role_date_conflict(candidate),
            "experience_exceeds_career_span": self.experience_exceeds_career_span(candidate, reference),
            "senior_title_pre_graduation": self.senior_title_pre_graduation(candidate),
            "num_education_overlaps": self.num_education_overlaps(candidate),
            "num_skill_anomalies": self.num_skill_anomalies(candidate),
            "num_proficiency_anomalies": self.num_proficiency_anomalies(candidate),
            "num_skill_anachronisms": self.num_skill_anachronisms(candidate, reference_frac),
            "skill_anachronism_years": self.skill_anachronism_years(candidate, reference_frac),
            "years_predating_company": self.years_predating_company(candidate),
        }
        # Return only the columns the policy declares, so the asset stays the source of truth.
        return {column: values[column] for column in self.columns}
