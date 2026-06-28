"""Compile the scoring policy into Polars expressions and score a feature frame.

The pipeline mirrors the policy's evaluator contract:

    base_tier        = clamp(sum(base additive flags) * product(internal gates), 0, 1)
    career_substance = base_tier + sum(bonus flags) * scale(base_tier)   # bonus un-capped
    skill_booster    = bonus when career_substance is high enough
    base_score       = career_substance + skill_booster   # lower-bounded at 0, NOT capped
    score            = base_score * product(multiplier stages)
                                  * product(integrity penalties) * product(hard gates)

The base tier is normalized to [0,1] ("meets the must-haves"); the bonus tier rides above
that cap so it differentiates must-have-complete candidates. The final score is a ranking
key, not a [0,1] probability -- it already exceeds 1.0 via the positive multiplier stages.

Integrity penalties are job-agnostic, small multiplicative data-quality factors that
compound: a genuine profile trips none, an implausible one trips several and slides down.
Every stage is expressed over columns, so scoring the whole pool is a single vectorized
pass. Each stage, penalty, and gate is kept as its own column for breakdown/debug.
"""

import polars as pl

from src.models.integrity import IntegrityPolicy
from src.models.policy import (
    CompositeProduct,
    Conditional,
    Curve,
    Decay,
    Lookup,
)
from src.models.tuning import Tuning
from src.ranking.predicate import compile_predicate

CAREER_SUBSTANCE = "career_substance"
SKILL_BOOSTER = "skill_booster"
BASE_SCORE = "base_score"
SCORE = "score"


def _multiplier_prefix(stage_id: str) -> str:
    return f"mult__{stage_id}"


def _gate_prefix(gate_id: str) -> str:
    return f"gate__{gate_id}"


def _lookup_expr(stage: Lookup) -> pl.Expr:
    return pl.col(stage.feature).replace_strict(
        stage.map, default=stage.default, return_dtype=pl.Float64
    )


def _curve_expr(stage: Curve) -> pl.Expr:
    feature = pl.col(stage.feature)
    bands = [(b.at, b.value) for b in stage.bands if b.at is not None]
    default = next((b.value for b in stage.bands if b.at is None), None)

    if stage.direction == "min":
        # Higher feature is better: first band (highest threshold) the value clears.
        bands.sort(key=lambda b: b[0], reverse=True)
        condition = lambda at: feature >= at
    else:
        # Lower feature is better: first band (lowest threshold) within reach.
        bands.sort(key=lambda b: b[0])
        condition = lambda at: feature <= at

    if default is None:
        default = bands[-1][1] if bands else 1.0

    expr = pl.when(condition(bands[0][0])).then(pl.lit(bands[0][1]))
    for at, value in bands[1:]:
        expr = expr.when(condition(at)).then(pl.lit(value))
    return expr.otherwise(pl.lit(default))


def _conditional_expr(stage: Conditional) -> pl.Expr:
    if not stage.cases:
        return pl.lit(stage.default)
    first = stage.cases[0]
    expr = pl.when(compile_predicate(first.when)).then(pl.lit(first.value))
    for case in stage.cases[1:]:
        expr = expr.when(compile_predicate(case.when)).then(pl.lit(case.value))
    return expr.otherwise(pl.lit(stage.default))


def _decay_expr(stage: Decay) -> pl.Expr:
    return pl.max_horizontal(
        pl.lit(stage.floor), pl.lit(stage.base).pow(pl.col(stage.feature))
    )


def _composite_expr(stage: CompositeProduct) -> pl.Expr:
    product = pl.lit(1.0)
    for member in stage.members:
        product = product * _stage_expr(member)
    low, high = stage.clamp
    return product.clip(low, high)


def _stage_expr(stage) -> pl.Expr:
    if isinstance(stage, Lookup):
        return _lookup_expr(stage)
    if isinstance(stage, Curve):
        return _curve_expr(stage)
    if isinstance(stage, Conditional):
        return _conditional_expr(stage)
    if isinstance(stage, Decay):
        return _decay_expr(stage)
    if isinstance(stage, CompositeProduct):
        return _composite_expr(stage)
    raise TypeError(f"Unsupported multiplier stage: {type(stage).__name__}")


def _tier_sum(weights: dict[str, float], requires) -> pl.Expr:
    """Sum a tier's flag weights, granting a flag's weight only when its (optional)
    `requires` predicate also holds. Shared by the base and bonus tiers."""
    return pl.sum_horizontal(
        [
            pl.when(
                pl.col(flag) & compile_predicate(requires[flag])
                if flag in requires
                else pl.col(flag)
            )
            .then(pl.lit(weight))
            .otherwise(0.0)
            for flag, weight in weights.items()
        ]
    )


def _career_substance_expr(tuning: Tuning) -> pl.Expr:
    cs = tuning.career_substance
    low, high = cs.clamp
    # Base tier: the JD's hard requirements, modulated by the ownership/domain gates.
    base = _tier_sum(cs.additive, cs.requires)
    for gate in cs.gates:
        base = base * pl.when(compile_predicate(gate.when)).then(pl.lit(gate.multiplier)).otherwise(1.0)
    base = base.clip(low, high)
    if cs.bonus is None:
        return base
    # Bonus tier: nice-to-haves credited in proportion to base strength (knee), so they
    # differentiate qualified candidates but never substitute for a missing core flag.
    # The bonus rides ABOVE the base clamp deliberately: the base axis saturates at `high`
    # (= "meets the must-haves"), and two must-have-complete engineers must still be
    # separated by how many JD differentiators they each bring. Re-clamping here would
    # erase exactly that separation at the top, where it matters most. The final score is
    # a ranking key, not a normalized [0,1] probability, so an un-capped sum is correct.
    bonus = _tier_sum(cs.bonus.additive, cs.requires)
    scale = (base / cs.bonus.knee).clip(0.0, 1.0)
    return base + bonus * scale


def _flag_columns(tuning: Tuning) -> list[str]:
    return list(tuning.features.flags.deterministic.keys()) + list(tuning.features.flags.slm)


def ceiling_expr(tuning: Tuning, integrity: IntegrityPolicy | None = None) -> pl.Expr:
    """Best-possible score for a candidate given only deterministic features.

    The multiplier stages, integrity penalties, and hard gates all depend on
    deterministic columns, so the only headroom is base_score (career_substance +
    skill_booster). A perfect SLM result maxes career_substance at base_sum + bonus_sum
    (the bonus rides above the base clamp), and the skill_booster adds its own max, so the
    best-possible base_score is computed from the policy rather than assumed to be 1.0.
    Holding base_score at that max and the gates at 1.0 (their no-penalty value) gives an
    upper bound on the achievable score: the multipliers and integrity penalties contribute
    their actual deterministic value. Pre-filtering on this never drops a candidate who
    could place.

    One stage (github_bonus) gates on career_substance, which the scorer computes
    rather than stores. The caller must provide a best-case `career_substance`
    column (at least 0.6) so that bonus can fire here; see `select_for_slm`.
    """
    cs = tuning.career_substance
    base_ceiling = min(sum(cs.additive.values()), cs.clamp[1])
    if cs.bonus is not None:
        base_ceiling += sum(cs.bonus.additive.values())
    base_ceiling += tuning.skill_booster.max
    ceiling = pl.lit(base_ceiling)
    for stage in tuning.multipliers:
        ceiling = ceiling * _stage_expr(stage)
    if integrity is not None:
        for stage in integrity.penalties:
            ceiling = ceiling * _stage_expr(stage)
    return ceiling


def score_frame(
    frame: pl.DataFrame, tuning: Tuning, integrity: IntegrityPolicy | None = None
) -> pl.DataFrame:
    """Return the frame with career_substance, per-stage, and final score columns."""
    # Uncertain handling: an undetermined flag contributes nothing and fires no
    # disqualifier, which for a boolean column means False.
    frame = frame.with_columns(
        [pl.col(flag).fill_null(False) for flag in _flag_columns(tuning)]
    )

    # Policy-driven flag overrides (career_substance's `derived_flags`): recompute selected
    # flags from a predicate over the freshly-landed columns, before they feed the base tier.
    # Expressed in the policy, not hardcoded here, so the engine stays JD-agnostic. The
    # canonical use is the Python rescue -- a career history almost never says "I used Python",
    # so the SLM's strong_python_prod under-fires; but owning a production retrieval/ranking/
    # eval system (or shipping end-to-end at scale) is itself proof of Python, so the policy
    # ORs those ownership signals in. Each override may stash its pre-override value under
    # `preserve_as` (e.g. strong_python_slm) for provenance in debug.jsonl / audit_trace.jsonl.
    # All expressions in one with_columns see the original frame, so `preserve_as` captures the
    # pre-override value even when `when` references the target itself.
    for spec in tuning.derived_flags:
        exprs: list[pl.Expr] = []
        if spec.preserve_as:
            exprs.append(pl.col(spec.target).alias(spec.preserve_as))
        exprs.append(compile_predicate(spec.when).alias(spec.target))
        frame = frame.with_columns(exprs)

    frame = frame.with_columns(_career_substance_expr(tuning).alias(CAREER_SUBSTANCE))

    booster = tuning.skill_booster
    booster_value = pl.min_horizontal(
        pl.lit(booster.max),
        pl.lit(booster.per_skill) * pl.col(booster.count_feature),
    )
    frame = frame.with_columns(
        pl.when(compile_predicate(booster.when)).then(booster_value).otherwise(0.0).alias(SKILL_BOOSTER)
    )
    # Lower bound only: career_substance (base + bonus) and the skill_booster are both
    # non-negative, and the sum is intentionally NOT re-capped at 1.0 -- the bonus tier
    # must keep differentiating must-have-complete candidates above the base saturation
    # point (see _career_substance_expr). The final score is a ranking key, not a [0,1]
    # probability, and already exceeds 1.0 via the positive (>1.0) multiplier stages.
    frame = frame.with_columns(
        (pl.col(CAREER_SUBSTANCE) + pl.col(SKILL_BOOSTER)).clip(lower_bound=0.0).alias(BASE_SCORE)
    )

    stage_columns: list[str] = []
    for stage in tuning.multipliers:
        name = _multiplier_prefix(stage.id or stage.type)
        frame = frame.with_columns(_stage_expr(stage).alias(name))
        stage_columns.append(name)

    # Job-agnostic integrity penalties: ordinary multiplier stages that compound.
    penalty_columns: list[str] = []
    if integrity is not None:
        for stage in integrity.penalties:
            name = _multiplier_prefix(stage.id or stage.type)
            frame = frame.with_columns(_stage_expr(stage).alias(name))
            penalty_columns.append(name)

    gate_columns: list[str] = []
    for gate in tuning.hard_gates:
        name = _gate_prefix(gate.id or "gate")
        frame = frame.with_columns(
            pl.when(compile_predicate(gate.when)).then(pl.lit(gate.multiplier)).otherwise(1.0).alias(name)
        )
        gate_columns.append(name)

    score = pl.col(BASE_SCORE)
    for name in stage_columns + penalty_columns + gate_columns:
        score = score * pl.col(name)
    return frame.with_columns(score.alias(SCORE))
