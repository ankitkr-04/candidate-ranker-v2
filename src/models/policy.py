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


class BonusTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Nice-to-have flags -- the JD's "Things we'd like but won't reject you for". Credited
    # in proportion to base strength: bonus * clamp(base / knee, 0, 1). At base >= knee the
    # full bonus applies (it differentiates already-qualified candidates, e.g. two ranking
    # owners where one also has distributed-systems depth); below knee the bonus is scaled
    # down so extras can never substitute for a missing core ("absolutely need") requirement.
    additive: dict[str, float]
    knee: float


class CareerSubstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Base tier: the JD's "Things you absolutely need" + the role mandate (own
    # retrieval/ranking, ship a system, evaluate it). This alone decides whether a
    # candidate meets the JD; the gates below and the bonus tier modulate it.
    additive: dict[str, float]
    # Optional second tier, gated on base strength so nice-to-haves lift the qualified
    # but cannot lift a candidate who lacks the core. See BonusTier.
    bonus: Optional[BonusTier] = None
    # Optional precondition per flag (base or bonus): the flag's credit is granted only
    # when its predicate also holds. Encodes logical dependencies the SLM can't be trusted
    # to enforce (e.g. owns_eval_framework / ltr_experience require an actual
    # ranking/retrieval system underneath), keeping the rule policy-driven not hardcoded.
    requires: dict[str, "Predicate"] = Field(default_factory=dict)
    gates: list[Gate] = Field(default_factory=list)
    clamp: tuple[float, float]


class SkillBooster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: "Predicate"
    per_skill: float
    max: float
    # Feature column holding the count the per_skill bonus multiplies. Named in the policy
    # (not hardcoded in the scorer) so the engine carries no this-job column names.
    count_feature: str = "num_qualifying_unevidenced_skills"
    credit_rule: str = ""
    qualifying: list[str] = Field(default_factory=list)
    disqualified: list[str] = Field(default_factory=list)


class HardGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    when: "Predicate"
    multiplier: float


class DerivedFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Recompute one feature flag from a predicate over the other (already-landed) columns,
    # applied after the deterministic + SLM columns are present but before scoring. This is
    # how the policy expresses logical rescues/overrides in *data* instead of engine code --
    # e.g. "credit strong_python_prod when the candidate owns a production retrieval/ranking/
    # eval system, since shipping that is itself proof of Python". `when` reuses the predicate
    # language and may reference `target` itself (it sees the pre-override value). `preserve_as`,
    # if set, keeps that pre-override value under a separate column for provenance/debug.
    target: str
    when: "Predicate"
    preserve_as: Optional[str] = None


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


class SubThresholdFloor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Ordering for candidates that bottom out at score 0 (no base/bonus flag fired). They are
    # re-spread by a deterministic substance proxy mapped strictly below the smallest positive
    # score, so a small CPU/--no-slm sample (the reproducibility sandbox) does not collapse the
    # unqualified tail onto the candidate_id tie-break. Omitted => disabled (the raw 0 tie-break
    # stands). The column names live here, not in the scorer, so the engine carries no
    # this-job feature names.
    #
    # The substance score sums two kinds of term:
    #   substance: numeric feature column -> weight (the column's value times the weight)
    #   categorical: categorical column -> {category value -> weight} (a lookup, unmatched = 0)
    # so ML-credited experience, raw experience, and title relevance can compose into one
    # ordering key. Both default empty; supply either or both.
    substance: dict[str, float] = Field(default_factory=dict)
    categorical: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Floored band ceiling = smallest positive score * headroom; < 1 keeps every floored
    # candidate strictly below every positive scorer, so any pool's ranked head is unchanged.
    headroom: float = 0.9


class Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uncertain_treatment: UncertainTreatment
    tie_break: list[str]
    sub_threshold_floor: Optional[SubThresholdFloor] = None


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
    # Optional policy-driven flag overrides applied before scoring (see DerivedFlag).
    derived_flags: list[DerivedFlag] = Field(default_factory=list)
    lookups: Lookups
    slm_questions: SlmQuestions


# Resolve forward references for the recursive Predicate / Multiplier types.
for _model in (NotNode, AllNode, AnyNode, Case, CompositeProduct, Gate, SkillBooster,
               HardGate, DerivedFlag):
    _model.model_rebuild()
