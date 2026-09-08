"""Evaluation cases for the grounded-answer pipeline: should it answer or abstain?"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PipelineCase(BaseModel):
    """One question and whether the pipeline is expected to abstain on it."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1)
    should_abstain: bool
    note: str = Field(min_length=1)


def load_pipeline_cases(path: str | Path) -> tuple[PipelineCase, ...]:
    """Load and validate a non-empty JSON array of pipeline cases with unique ids."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("pipeline evaluation file must contain a JSON array")

    cases = tuple(PipelineCase.model_validate(item) for item in raw)
    if not cases:
        raise ValueError("pipeline evaluation file must contain at least one case")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("pipeline evaluation case_id values must be unique")
    return cases
