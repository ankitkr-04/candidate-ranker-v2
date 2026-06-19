"""String normalization for matching free-text profile fields to policy lookups.

The policy's company, location, and title lookups are canonical lowercase
strings. These helpers reduce noisy profile values to the same form so that
membership checks are reliable. The alias maps only need to cover spelling
variants the lookups do not already list both forms of.
"""

import re

_WHITESPACE = re.compile(r"\s+")

_CITY_ALIASES = {
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
    "new delhi": "delhi",
}

_COMPANY_ALIASES = {
    "tata consultancy services": "tcs",
    "byjus": "byju's",
}


def normalize_token(value: str) -> str:
    """Lowercase, trim, and collapse internal whitespace."""
    return _WHITESPACE.sub(" ", (value or "").strip().lower())


def normalize_city(location: str) -> str:
    """Reduce a "City, Region" location string to a canonical city name."""
    city = normalize_token(location).split(",")[0].strip()
    return _CITY_ALIASES.get(city, city)


def normalize_company(name: str) -> str:
    company = normalize_token(name)
    return _COMPANY_ALIASES.get(company, company)


def normalize_title(title: str) -> str:
    return normalize_token(title)
