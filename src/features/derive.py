"""Lookup-driven feature derivation: flags, categoricals, and ML-credit metrics.

These derivations depend on the policy's company, location, and title lookups,
so the deriver is constructed from a Tuning instance and precomputes normalized
lookup sets once, then reuses them across the whole candidate pool.
"""

from src.features.normalize import (
    normalize_city,
    normalize_company,
    normalize_title,
    normalize_token,
)
from src.models.candidate import Candidate
from src.models.integrity import SeniorityLadder
from src.models.tuning import Tuning

_INDIA = "india"

# Single source of truth for the seniority ladder is the job-agnostic integrity config
# (SeniorityLadder); this module-level instance is the fallback used when no ladder is
# supplied, and reproduces the historical hardcoded values.
_DEFAULT_LADDER = SeniorityLadder()


def _seniority_rank(title: str, ladder: SeniorityLadder = _DEFAULT_LADDER) -> int:
    # Tiers are matched in order with later tiers overriding earlier ones, so
    # "engineering manager" ranks above "senior engineer".
    padded = f" {normalize_title(title)} "
    rank = ladder.default
    for tier in ladder.tiers:
        if any(f" {k} " in padded for k in tier.keywords):
            rank = tier.rank
    return rank


class FeatureDeriver:
    def __init__(self, tuning: Tuning, seniority_ladder: SeniorityLadder | None = None) -> None:
        self.tuning = tuning
        self.slm_flag_names = list(tuning.features.flags.slm)
        # Shared seniority ladder (from the job-agnostic integrity config); falls back to the
        # default when precompute runs without an integrity policy.
        self._seniority_ladder = seniority_ladder or _DEFAULT_LADDER

        company = tuning.lookups.company
        self._it_services = {normalize_company(x) for x in company.it_services}
        self._ai_native = {normalize_company(x) for x in company.ai_native}
        # product_set names other categories rather than companies; expand them.
        category_members = {
            "it_services": company.it_services,
            "fictional_neutral": company.fictional_neutral,
            "product_india": company.product_india,
            "ai_native": company.ai_native,
            "tier_0": company.tier_0,
        }
        self._product: set[str] = set()
        for category in company.product_set:
            self._product.update(normalize_company(x) for x in category_members.get(category, []))

        location = tuning.lookups.location
        self._local_cities = {normalize_city(x) for x in location.local_cities}
        self._commutable_cities = {normalize_city(x) for x in location.commutable_cities}
        self._tier1_cities = {normalize_city(x) for x in location.tier1_cities}

        title = tuning.lookups.title
        self._title_bucket: dict[str, str] = {}
        for bucket, titles in title.current_title_buckets.items():
            for name in titles:
                self._title_bucket[normalize_title(name)] = bucket

        self._title_factor: dict[str, float] = {}
        for tier in title.career_ml_credit.values():
            for name in tier.titles:
                self._title_factor[normalize_title(name)] = tier.factor

        self._qualifying_skills = {normalize_token(s) for s in tuning.skill_booster.qualifying}

    # Company-based flags ----------------------------------------------------

    def current_is_services(self, c: Candidate) -> bool:
        return normalize_company(c.profile.current_company) in self._it_services

    def has_ai_native(self, c: Candidate) -> bool:
        return any(normalize_company(r.company) in self._ai_native for r in c.career_history)

    def has_product_company(self, c: Candidate) -> bool:
        return any(normalize_company(r.company) in self._product for r in c.career_history)

    def majority_career_services(self, c: Candidate) -> bool:
        total = sum(r.duration_months for r in c.career_history)
        if total <= 0:
            return False
        services = sum(
            r.duration_months
            for r in c.career_history
            if normalize_company(r.company) in self._it_services
        )
        return (services / total) > 0.5

    def enterprise_lifer(self, c: Candidate) -> bool:
        # The JD warns against whole-career big-enterprise tenure ("if you've spent
        # your career at Google or Meta ... this isn't it"). Fire only when every role
        # is at a 10001+ employer and there are at least two, so a single big-company
        # stint among smaller ones does not trip it.
        roles = c.career_history
        if len(roles) < 2:
            return False
        return all(r.company_size == "10001+" for r in roles)

    # Location / title flags -------------------------------------------------

    def is_local(self, c: Candidate) -> bool:
        return normalize_city(c.profile.location) in self._local_cities

    def titles_escalating(self, c: Candidate) -> bool:
        recent = c.career_history[:3]
        if len(recent) < 2:
            return False
        ranks = [_seniority_rank(r.title, self._seniority_ladder) for r in reversed(recent)]  # oldest -> newest
        return all(later > earlier for earlier, later in zip(ranks, ranks[1:]))

    # Metrics / categoricals -------------------------------------------------

    def applied_ml_years(self, c: Candidate) -> float:
        credited = sum(
            (r.duration_months / 12.0) * self._title_factor.get(normalize_title(r.title), 0.0)
            for r in c.career_history
        )
        return min(c.profile.years_of_experience, credited)

    def num_qualifying_unevidenced_skills(self, c: Candidate) -> float:
        # The competence-flag exclusion from the policy depends on SLM outputs and
        # is applied when those are merged; the base count is the qualifying skills
        # the candidate lists.
        listed = {normalize_token(s.name) for s in c.skills}
        return float(len(listed & self._qualifying_skills))

    def current_title_bucket(self, c: Candidate) -> str:
        # Unmatched titles fall to the neutral bucket, which the multiplier maps to
        # the same value as its default.
        return self._title_bucket.get(normalize_title(c.profile.current_title), "neutral_read_description")

    def location_relocation_bucket(self, c: Candidate) -> str:
        city = normalize_city(c.profile.location)
        if city in self._local_cities:
            return "local"
        if normalize_token(c.profile.country) == _INDIA:
            if city in self._commutable_cities:
                base = "commutable"
            elif city in self._tier1_cities:
                base = "tier1"
            else:
                base = "other_india"
        else:
            base = "outside"
        # Location fit is governed by where the candidate is based and whether they will
        # relocate -- not by work-mode preference. The JD's "async-first / no required
        # in-office days" describes working *style* (it lives in the vibe check), not a
        # licence to live anywhere: the logistics section names a deliberate exempt-from-
        # relocation whitelist (Hyderabad, Pune, Mumbai, Delhi NCR) and pointedly omits
        # Bangalore/Chennai. A remote preference does not substitute for being in an exempt
        # city or agreeing to relocate.
        suffix = "relocating" if c.redrob_signals.willing_to_relocate else "not_relocating"
        return f"{base}_{suffix}"

    def verification_state(self, c: Candidate) -> str:
        verified = int(c.redrob_signals.verified_email) + int(c.redrob_signals.verified_phone)
        return {2: "both", 1: "one", 0: "neither"}[verified]
