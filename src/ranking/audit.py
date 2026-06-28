"""Human-readable scoring audit trace.

`debug.jsonl` dumps the scored frame flat -- every feature and every per-stage
factor as a separate column -- which answers "what are the values" but not "how did
those values produce this score". This module re-renders the exact same numbers as an
ordered derivation: base build-up, then each multiplier / integrity penalty / hard gate
that actually moved the score, with the running product after each step and a one-line
formula. It re-scores nothing; it only reads the columns `score_frame` already wrote, so
the trace is faithful by construction and self-checks against the stored `score`.
"""

from __future__ import annotations

from src.models.integrity import IntegrityPolicy
from src.models.tuning import Tuning
from src.ranking.scorer import (
    BASE_SCORE,
    CAREER_SUBSTANCE,
    SCORE,
    SKILL_BOOSTER,
    _gate_prefix,
    _multiplier_prefix,
)

_EPS = 1e-9


def _pct(factor: float) -> str:
    return f"{(factor - 1.0) * 100:+.1f}%"


def _stages(tuning: Tuning, integrity: IntegrityPolicy | None) -> list[tuple[str, str, str]]:
    """Ordered (stage_id, column, source) tuples matching the scoring multiplication order."""
    order: list[tuple[str, str, str]] = []
    for stage in tuning.multipliers:
        sid = stage.id or stage.type
        order.append((sid, _multiplier_prefix(sid), "multiplier"))
    if integrity is not None:
        for stage in integrity.penalties:
            sid = stage.id or stage.type
            order.append((sid, _multiplier_prefix(sid), "integrity"))
    for gate in tuning.hard_gates:
        sid = gate.id or "gate"
        order.append((sid, _gate_prefix(sid), "gate"))
    return order


def trace_row(row: dict, stages: list[tuple[str, str, str]]) -> dict:
    """One candidate's scoring derivation, showing only the stages that moved the score."""
    base_cs = float(row[CAREER_SUBSTANCE])
    booster = float(row[SKILL_BOOSTER])
    base = float(row[BASE_SCORE])

    running = base
    steps: list[dict] = []
    neutral: list[str] = []
    formula = [f"({base_cs:.3f} substance + {booster:.3f} booster) = {base:.4f}"]

    for stage_id, column, source in stages:
        factor = float(row[column])
        running *= factor
        if abs(factor - 1.0) <= _EPS:
            neutral.append(stage_id)
            continue
        kind = "bonus" if factor > 1.0 else ("gate" if source == "gate" else "penalty")
        steps.append(
            {
                "stage": stage_id,
                "source": source,
                "kind": kind,
                "factor": round(factor, 6),
                "effect": _pct(factor),
                "running": round(running, 6),
            }
        )
        formula.append(f"x {factor:.3f} ({stage_id})")

    score = float(row[SCORE])
    return {
        "rank": row.get("rank"),
        "candidate_id": row.get("candidate_id"),
        "score": round(score, 6),
        "base": {
            "career_substance": round(base_cs, 6),
            "skill_booster": round(booster, 6),
            "base_score": round(base, 6),
        },
        "steps": steps,
        "neutral_stages": neutral,
        "formula": "  ".join(formula) + f"  = {running:.4f}",
        "check": {"recomputed": round(running, 6), "matches": abs(running - score) <= 1e-6},
    }


def build_audit_trace(
    rows: list[dict], tuning: Tuning, integrity: IntegrityPolicy | None
) -> list[dict]:
    stages = _stages(tuning, integrity)
    return [trace_row(row, stages) for row in rows]
