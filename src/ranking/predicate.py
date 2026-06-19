"""Compile the policy's predicate language into Polars boolean expressions.

Each `when` clause in the policy is a small tree of flag/metric leaves and
all/any/not nodes. Compiling it to a `pl.Expr` lets the whole pool be evaluated
in one vectorized pass instead of a per-row Python loop.

Flag columns are expected to be non-null before compilation (the scorer fills
SLM flags with False per the policy's uncertain handling), so a flag leaf maps
directly to its boolean column.
"""

import operator
from typing import Callable

import polars as pl

from src.models.policy import AllNode, AnyNode, FlagLeaf, MetricLeaf, NotNode, Predicate

_COMPARATORS: dict[str, Callable] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def compile_predicate(pred: Predicate) -> pl.Expr:
    if isinstance(pred, FlagLeaf):
        column = pl.col(pred.flag)
        return ~column if pred.negate else column
    if isinstance(pred, MetricLeaf):
        return _COMPARATORS[pred.op](pl.col(pred.metric), pred.value)
    if isinstance(pred, NotNode):
        return ~compile_predicate(pred.operand)
    if isinstance(pred, AllNode):
        return pl.all_horizontal([compile_predicate(p) for p in pred.all])
    if isinstance(pred, AnyNode):
        return pl.any_horizontal([compile_predicate(p) for p in pred.any])
    raise TypeError(f"Unsupported predicate node: {type(pred).__name__}")
