"""Schema of the precomputed feature table (features.parquet).

The column set is derived from the policy so it cannot drift from what the scorer
references: flags come from features.flags, metrics from features.metrics (minus
career_substance, which the scorer computes during ranking), and categoricals
from features.categoricals. A few raw display fields and the SLM text outputs
complete the table. Every column is flat and typed for vectorized scoring.
"""

import polars as pl

from src.models.tuning import Tuning

# Produced during scoring rather than precomputed.
_SCORER_COMPUTED_METRICS = {"career_substance"}

DISPLAY_COLUMNS = [
    "current_title",
    "current_company",
    "location",
    "country",
    "preferred_work_mode",
    "willing_to_relocate",
]
SLM_TEXT_COLUMNS = ["subject_of_primary_work", "evidence"]


def deterministic_flag_columns(tuning: Tuning) -> list[str]:
    return list(tuning.features.flags.deterministic.keys())


def slm_flag_columns(tuning: Tuning) -> list[str]:
    return list(tuning.features.flags.slm)


def metric_columns(tuning: Tuning) -> list[str]:
    return [m for m in tuning.features.metrics if m not in _SCORER_COMPUTED_METRICS]


def categorical_columns(tuning: Tuning) -> list[str]:
    return list(tuning.features.categoricals.keys())


def parquet_schema(tuning: Tuning) -> dict[str, pl.DataType]:
    """Ordered column -> dtype mapping for the feature table."""
    string_dtype = pl.String()
    boolean_dtype = pl.Boolean()
    float_dtype = pl.Float64()

    schema: dict[str, pl.DataType] = {"candidate_id": string_dtype}
    for flag in deterministic_flag_columns(tuning):
        schema[flag] = boolean_dtype
    for flag in slm_flag_columns(tuning):
        schema[flag] = boolean_dtype
    for metric in metric_columns(tuning):
        schema[metric] = float_dtype
    for categorical in categorical_columns(tuning):
        schema[categorical] = string_dtype
    for display in DISPLAY_COLUMNS:
        schema[display] = boolean_dtype if display == "willing_to_relocate" else string_dtype
    for text in SLM_TEXT_COLUMNS:
        schema[text] = string_dtype
    return schema
