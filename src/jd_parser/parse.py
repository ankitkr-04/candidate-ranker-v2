"""Validate the parsed job description and emit the downstream tuning artifacts.

Loads assets/job/jd_parsed.json, validates it against the typed `Policy` model
(the JD-drift early warning), then writes:
  - artifacts/tuning/tuning.json          ranker + feature-build knobs
  - artifacts/tuning/slm_questions.json   the question set for the SLM

Run with: python -m src.jd_parser.parse
"""

import json
from pathlib import Path

from src.models.policy import Policy
from src.models.tuning import Tuning
from src.paths import JOB_DIR, TUNING_ARTIFACT_DIR


def load_policy(path: Path | None = None) -> Policy:
    """Load and validate the parsed job description into a typed Policy."""
    source = path or (JOB_DIR / "jd_parsed.json")
    data = json.loads(source.read_text())
    return Policy.model_validate(data)


def write_artifacts(policy: Policy, out_dir: Path = TUNING_ARTIFACT_DIR) -> tuple[Path, Path]:
    """Write tuning.json and slm_questions.json; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = out_dir / "tuning.json"
    questions_path = out_dir / "slm_questions.json"

    tuning = Tuning.from_policy(policy)
    # by_alias preserves reserved keys such as the predicate "not"; exclude_none
    # drops the many optional `id` fields that are absent in the source.
    tuning_path.write_text(tuning.model_dump_json(by_alias=True, exclude_none=True, indent=2))
    questions_path.write_text(
        policy.slm_questions.model_dump_json(by_alias=True, exclude_none=True, indent=2)
    )
    return tuning_path, questions_path


def main() -> None:
    policy = load_policy()
    tuning_path, questions_path = write_artifacts(policy)
    print(f"Validated policy {policy.policy_version} ({policy.job_id})")
    print(f"  wrote {tuning_path}")
    print(f"  wrote {questions_path}")


if __name__ == "__main__":
    main()
