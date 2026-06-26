"""Compile the scoring policy into Polars expressions and score a feature frame.

The pipeline mirrors the policy's evaluator contract:

    career_substance = clamp(sum(additive flags) * product(internal gates))
    skill_booster    = bonus when career_substance is high enough
    base_score       = clamp(career_substance + skill_booster, 0, 1)
    score            = base_score * product(multiplier stages)
                                  * product(integrity penalties) * product(hard gates)

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


def _career_substance_expr(tuning: Tuning) -> pl.Expr:
    cs = tuning.career_substance
    additive = pl.sum_horizontal(
        [
            pl.when(
                pl.col(flag) & compile_predicate(cs.requires[flag])
                if flag in cs.requires
                else pl.col(flag)
            )
            .then(pl.lit(weight))
            .otherwise(0.0)
            for flag, weight in cs.additive.items()
        ]
    )
    expr = additive
    for gate in cs.gates:
        expr = expr * pl.when(compile_predicate(gate.when)).then(pl.lit(gate.multiplier)).otherwise(1.0)
    low, high = cs.clamp
    return expr.clip(low, high)


def _flag_columns(tuning: Tuning) -> list[str]:
    return list(tuning.features.flags.deterministic.keys()) + list(tuning.features.flags.slm)


def ceiling_expr(tuning: Tuning, integrity: IntegrityPolicy | None = None) -> pl.Expr:
    """Best-possible score for a candidate given only deterministic features.

    The multiplier stages, integrity penalties, and hard gates all depend on
    deterministic columns, so the only headroom is base_score (career_substance +
    skill_booster), which a perfect SLM result maxes at 1.0. Holding base_score and
    the gates at 1.0 (their no-penalty value) gives an upper bound on the achievable
    score: the multipliers and integrity penalties contribute their actual
    deterministic value. Pre-filtering on this never drops a candidate who could place.

    One stage (github_bonus) gates on career_substance, which the scorer computes
    rather than stores. The caller must provide a best-case `career_substance`
    column (1.0) so that bonus can fire here; see `select_for_slm`.
    """
    ceiling = pl.lit(1.0)
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

    frame = frame.with_columns(_career_substance_expr(tuning).alias(CAREER_SUBSTANCE))

    booster = tuning.skill_booster
    booster_value = pl.min_horizontal(
        pl.lit(booster.max),
        pl.lit(booster.per_skill) * pl.col("num_qualifying_unevidenced_skills"),
    )
    frame = frame.with_columns(
        pl.when(compile_predicate(booster.when)).then(booster_value).otherwise(0.0).alias(SKILL_BOOSTER)
    )
    frame = frame.with_columns(
        (pl.col(CAREER_SUBSTANCE) + pl.col(SKILL_BOOSTER)).clip(0.0, 1.0).alias(BASE_SCORE)
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
