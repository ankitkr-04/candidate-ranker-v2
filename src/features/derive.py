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
from src.models.tuning import Tuning

_INDIA = "india"

# Title keywords mapped to a coarse seniority rank. Later entries override
# earlier ones, so "engineering manager" ranks above "senior engineer".
_SENIORITY_TIERS = [
    ((" junior ", " jr ", " associate ", " trainee "), 1),
    ((" senior ", " sr ", " lead ", " principal ", " staff "), 3),
    ((" manager ", " head ", " director ", " vp ", " chief ", " cto "), 4),
]
_DEFAULT_SENIORITY = 2


def _seniority_rank(title: str) -> int:
    padded = f" {normalize_title(title)} "
    rank = _DEFAULT_SENIORITY
    if " intern " in padded:
        rank = 0
    for keywords, value in _SENIORITY_TIERS:
        if any(k in padded for k in keywords):
            rank = value
    return rank


class FeatureDeriver:
    def __init__(self, tuning: Tuning) -> None:
        self.tuning = tuning
        self.slm_flag_names = list(tuning.features.flags.slm)

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
        ranks = [_seniority_rank(r.title) for r in reversed(recent)]  # oldest -> newest
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
            base = "tier1" if city in self._tier1_cities else "other_india"
        else:
            base = "outside"
        suffix = "relocating" if c.redrob_signals.willing_to_relocate else "not_relocating"
        return f"{base}_{suffix}"

    def verification_state(self, c: Candidate) -> str:
        verified = int(c.redrob_signals.verified_email) + int(c.redrob_signals.verified_phone)
        return {2: "both", 1: "one", 0: "neither"}[verified]
