"""Typed model of the job scoring policy (assets/job/jd_parsed.json).

The policy is validated into these models so that any change to the released job
description that the code does not understand fails loudly at parse time rather
than silently mis-scoring. Two structures carry most of the policy's logic:

  - Predicate: the recursive boolean language used by `when` clauses. Each form
    (flag / metric / not / all / any) is its own model with `extra="forbid"`, so a
    raw dict resolves unambiguously to exactly one variant.
  - Multiplier: a union of the five multiplier stage types, discriminated by the
    `type` field; `composite_product` nests further multipliers.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

ComparisonOp = Literal["<", "<=", ">", ">=", "==", "!="]


# Predicate language ---------------------------------------------------------

class FlagLeaf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: str
    negate: bool = False


class MetricLeaf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    op: ComparisonOp
    value: float


class NotNode(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    operand: "Predicate" = Field(alias="not")


class AllNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all: list["Predicate"]


class AnyNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    any: list["Predicate"]


Predicate = Union[FlagLeaf, MetricLeaf, NotNode, AllNode, AnyNode]


# Multiplier stages ----------------------------------------------------------

class Band(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: Optional[float] = None
    value: float


class Lookup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lookup"]
    id: Optional[str] = None
    feature: str
    map: dict[str, float]
    default: float


class Curve(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["curve"]
    id: Optional[str] = None
    feature: str
    direction: Literal["min", "max"]
    bands: list[Band]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: "Predicate"
    value: float
    id: Optional[str] = None


class Conditional(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["conditional"]
    id: Optional[str] = None
    default: float
    cases: list[Case] = Field(default_factory=list)


class Decay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["decay"]
    id: Optional[str] = None
    feature: str
    base: float
    floor: float


class CompositeProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["composite_product"]
    id: Optional[str] = None
    clamp: tuple[float, float]
    members: list["Multiplier"]


Multiplier = Annotated[
    Union[Lookup, Curve, Conditional, Decay, CompositeProduct],
    Field(discriminator="type"),
]


# Scoring blocks -------------------------------------------------------------

class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: "Predicate"
    multiplier: float
    id: Optional[str] = None


class CareerSubstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    additive: dict[str, float]
    # Optional precondition per additive flag: the flag's credit is granted only when
    # its predicate also holds. Encodes logical dependencies the SLM can't be trusted
    # to enforce (e.g. owns_eval_framework requires an actual ranking/retrieval system
    # to evaluate), keeping the rule policy-driven rather than hardcoded in the scorer.
    requires: dict[str, "Predicate"] = Field(default_factory=dict)
    gates: list[Gate] = Field(default_factory=list)
    clamp: tuple[float, float]


class SkillBooster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: "Predicate"
    per_skill: float
    max: float
    credit_rule: str = ""
    qualifying: list[str] = Field(default_factory=list)
    disqualified: list[str] = Field(default_factory=list)


class HardGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    when: "Predicate"
    multiplier: float


# Lookups --------------------------------------------------------------------

class CompanyLookups(BaseModel):
    # extra allowed so new company categories in the JD do not break parsing.
    model_config = ConfigDict(extra="allow")

    it_services: list[str] = Field(default_factory=list)
    fictional_neutral: list[str] = Field(default_factory=list)
    product_india: list[str] = Field(default_factory=list)
    ai_native: list[str] = Field(default_factory=list)
    tier_0: list[str] = Field(default_factory=list)
    # Names of other categories whose members count as "product" companies.
    product_set: list[str] = Field(default_factory=list)


class LocationLookups(BaseModel):
    model_config = ConfigDict(extra="allow")

    local_cities: list[str] = Field(default_factory=list)
    # Tier-1 cities the JD names as "welcome to apply" without relocating (reachable for
    # quarterly offsites): Hyderabad, Mumbai. Distinct from tier1_cities (Bangalore/Chennai),
    # which the JD treats as relocation-preferred.
    commutable_cities: list[str] = Field(default_factory=list)
    tier1_cities: list[str] = Field(default_factory=list)


class CareerMlCreditTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: float
    titles: list[str]


class TitleLookups(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_title_buckets: dict[str, list[str]]
    career_ml_credit: dict[str, CareerMlCreditTier]


class Lookups(BaseModel):
    model_config = ConfigDict(extra="allow")

    company: CompanyLookups
    location: LocationLookups
    title: TitleLookups


# Top-level policy sections --------------------------------------------------

class UncertainTreatment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_flag_cannot_determine: str
    disqualifier_flag_cannot_determine: str


class Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uncertain_treatment: UncertainTreatment
    tie_break: list[str]


class FlagDefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slm: list[str]
    deterministic: dict[str, str]


class CategoricalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derive: str
    values: list[str]


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: FlagDefs
    metrics: dict[str, str]
    categoricals: dict[str, CategoricalSpec]


class SlmQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    q: str
    type: Optional[str] = None


class SlmQuestions(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_scope: str
    extraction_order: str
    ask: list[SlmQuestion]


class EvaluatorContract(BaseModel):
    # Documentation block; kept permissive since it is descriptive, not executable.
    model_config = ConfigDict(extra="allow")

    predicate: str = ""
    multiplier_stage_types: dict[str, str] = Field(default_factory=dict)
    pipeline: str = ""


class Policy(BaseModel):
    # Forbid unknown top-level keys: a new section in the JD should be a conscious
    # code change, not a silently ignored field. populate_by_name lets $schema alias work.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str | None = Field(None, alias="$schema")

    policy_version: str
    job_id: str
    evaluator_contract: EvaluatorContract
    scoring: Scoring
    features: Features
    # Plausibility/data-quality rules (date-consistency checks, anomaly penalties) are
    # job-agnostic and live in the separate integrity penalty layer, not the JD.
    career_substance: CareerSubstance
    skill_booster: SkillBooster
    multipliers: list[Multiplier]
    hard_gates: list[HardGate]
    lookups: Lookups
    slm_questions: SlmQuestions


# Resolve forward references for the recursive Predicate / Multiplier types.
for _model in (NotNode, AllNode, AnyNode, Case, CompositeProduct, Gate, SkillBooster,
               HardGate):
    _model.model_rebuild()
