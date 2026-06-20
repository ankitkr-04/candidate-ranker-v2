# Integrity layer (job-agnostic plausibility)

A separate, hand-authored layer of deterministic penalties for data that is implausible for
*any* genuine candidate, regardless of the role — fabricated timelines, anachronistic
skills, a senior title dated before the first degree finished.

## Why it is separate from the JD

The dividing line: a signal lives here only if it is **wrong for a genuine candidate**.
Anything that could legitimately occur — company/location preferences, work mode, title
trajectory — stays in the JD tuning. Keeping the two apart means a different job reuses this
layer untouched, and editing the JD never disturbs these checks.

```
assets/integrity/penalties.json          (source; hand-authored, job-agnostic)
        |  python -m src.jd_parser.parse  (validates -> writes artifact)
        v
artifacts/tuning/integrity.json           (generated; both stages read THIS)
```

It reuses the policy's `Multiplier` / `Predicate` schema, so the same models validate it and
the same scorer compiles it — no new format. The penalties are applied as ordinary
multiplier stages.

## Signals

Computed in `src/features/integrity.py` from data already in the Parquet inputs; all
thresholds come from the source asset (`params`, `tool_eras`), nothing is hardcoded.

- Date consistency: an end date before its start; a current-role date conflict.
- Tenure overruns: total or single-role months exceeding stated experience (plus a slack of
  `overrun_slack_months`, default 18).
- `num_education_overlaps`: overlapping degree spans.
- `num_skill_anomalies`: a skill claimed for more months than the candidate's experience.
- `num_skill_anachronisms`: a skill whose implied start year precedes the year the tool
  plausibly existed (`tool_eras`, e.g. prompt engineering -> 2020).
- `senior_title_pre_graduation`: a senior-ranked title starting before the **earliest**
  degree finished (baselining on the first degree avoids penalising a later part-time degree
  taken while already senior).

## How penalties combine

Several small, independent penalties that **compound multiplicatively**. A genuine candidate
trips none (x1.0); a fabricated profile trips several and slides down past the real people.
There is no single brittle hard-cut keyword — data-quality is handled entirely as this
graduated penalty gradient. The only hard zero is the date-impossibility honeypot (`hp_*`
flags), which is a side effect of the same checks, never a special case.

Fired penalties surface in the reasoning as specific concerns ("claims a skill predating the
technology", "a senior title dated before the first degree finished"), not as a label.

## Tuning

Edit a multiplier value or a `tool_eras` entry in `assets/integrity/penalties.json`, re-run
`python -m src.jd_parser.parse`, then re-rank (seconds; no precompute, no GPU). For a quick
experiment you can edit `artifacts/tuning/integrity.json` directly and re-rank.
