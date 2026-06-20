"""Job-agnostic profile-integrity penalties.

These signals flag data that is implausible for any genuine candidate regardless of the
role: inconsistent employment dates, a skill claimed for longer than the person has worked
or longer than the tool has existed, a senior title dated before the first degree finished.
They are not job-specific (unlike the JD's company/location/role-fit knobs), so they live in
their own asset (`assets/integrity/penalties.json`) and are parsed into their own artifact
(`artifacts/tuning/integrity.json`), leaving the JD tuning untouched.

The penalties reuse the policy's Multiplier/Predicate schema, so the same models validate
them and the same scorer compiles them. They are applied as ordinary multiplier stages:
several small penalties compound into a ranking gradient rather than one brittle hard cut.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.models.policy import Multiplier
from src.paths import INTEGRITY_ARTIFACT, INTEGRITY_SOURCE


class IntegrityParams(BaseModel):
    """Tunable thresholds for the integrity feature computations (no hardcoding in code)."""

    model_config = ConfigDict(extra="forbid")

    # Slack added to years_of_experience (in months) before total or single-role tenure
    # counts as an overrun; absorbs rounding and brief, legitimate role overlaps.
    overrun_slack_months: float = 18.0
    # Minimum seniority rank (see features.derive._seniority_rank) that reads as "senior".
    seniority_min_rank: int = 3


class IntegrityFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class IntegrityPolicy(BaseModel):
    """The parsed integrity layer: tunable params, declared features, and penalty stages."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str | None = Field(None, alias="$schema")
    version: str
    description: str = ""
    # Normalized skill/tool name -> earliest year the tool plausibly existed. A skill whose
    # implied start year precedes this is anachronistic.
    tool_eras: dict[str, int] = Field(default_factory=dict)
    params: IntegrityParams = Field(default_factory=IntegrityParams)
    features: IntegrityFeatures
    penalties: list[Multiplier] = Field(default_factory=list)


def load_integrity_source(path: Path | None = None) -> IntegrityPolicy:
    """Validate the hand-authored source asset (used by the parser)."""
    source = path or INTEGRITY_SOURCE
    return IntegrityPolicy.model_validate_json(source.read_text())


def load_integrity(path: Path | None = None) -> IntegrityPolicy:
    """Load the generated artifact consumed by precompute and the ranker."""
    artifact = path or INTEGRITY_ARTIFACT
    if not artifact.is_file():
        raise FileNotFoundError(
            f"{artifact} not found. Run `python -m src.jd_parser.parse` first."
        )
    return IntegrityPolicy.model_validate_json(artifact.read_text())
