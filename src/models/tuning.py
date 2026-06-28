"""Ranker-facing tuning artifact derived from the full policy.

`Tuning` is everything the feature-build and ranker need to score candidates: the
numeric knobs, gates, multipliers, and lookups. It deliberately omits the SLM
question prose (carried separately as `SlmQuestionSet`) so the scoring config and
the model-prompting config can be regenerated and versioned independently.
"""

from pydantic import BaseModel

from src.models.policy import (
    CareerSubstance,
    DerivedFlag,
    Features,
    HardGate,
    Lookups,
    Multiplier,
    Policy,
    Scoring,
    SkillBooster,
    SlmQuestions,
)

# The extracted question set is structurally identical to the policy's block.
SlmQuestionSet = SlmQuestions


class Tuning(BaseModel):
    policy_version: str
    job_id: str
    scoring: Scoring
    features: Features
    career_substance: CareerSubstance
    skill_booster: SkillBooster
    multipliers: list[Multiplier]
    hard_gates: list[HardGate]
    derived_flags: list[DerivedFlag] = []
    lookups: Lookups

    @classmethod
    def from_policy(cls, policy: Policy) -> "Tuning":
        return cls(
            policy_version=policy.policy_version,
            job_id=policy.job_id,
            scoring=policy.scoring,
            features=policy.features,
            career_substance=policy.career_substance,
            skill_booster=policy.skill_booster,
            multipliers=policy.multipliers,
            hard_gates=policy.hard_gates,
            derived_flags=policy.derived_flags,
            lookups=policy.lookups,
        )
