"""Run-level configuration for the current dataset snapshot.

These are properties of *this* candidate pool / run, not of the job policy (which lives in
`assets/job/jd_parsed.json`) nor of the job-agnostic integrity layer
(`assets/integrity/penalties.json`). Keep dataset-snapshot constants here so the rest of the
pipeline never reaches for the wall clock.
"""

from __future__ import annotations

from datetime import date

# Wall-clock anchor for date math (total-experience reads are static fields, but recency and
# the anachronism reference-year derive from the pool's most-recent last_active_date). That
# data-driven reference is used whenever the pool carries activity dates; this anchor is only
# the fallback for an empty/date-less pool, pinned so a run is reproducible-by-config instead
# of depending on date.today(). It reflects the snapshot date of the current dataset.
AS_OF_DATE = date(2026, 6, 30)
