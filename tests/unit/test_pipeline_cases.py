"""Tests for the pipeline evaluation-case loader."""

import json
from pathlib import Path

import pytest

from turbofan_copilot.evaluation.pipeline_cases import PipelineCase, load_pipeline_cases

REAL_CASES = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "pipeline_cases.json"


def write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def one_case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "case_id": "example",
        "question": "How does a chip detector work?",
        "should_abstain": False,
        "note": "Covered by Chapter 6.",
    }
    case.update(overrides)
    return case


def test_real_cases_load_with_a_mix_of_answer_and_abstain() -> None:
    cases = load_pipeline_cases(REAL_CASES)

    assert len(cases) >= 6
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(case.should_abstain for case in cases)
    assert any(not case.should_abstain for case in cases)


def test_loader_rejects_a_non_array(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain a JSON array"):
        load_pipeline_cases(write(tmp_path / "cases.json", {"case_id": "x"}))


def test_loader_rejects_an_empty_array(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one case"):
        load_pipeline_cases(write(tmp_path / "cases.json", []))


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    payload = [one_case(case_id="dup"), one_case(case_id="dup", question="another?")]
    with pytest.raises(ValueError, match="case_id values must be unique"):
        load_pipeline_cases(write(tmp_path / "cases.json", payload))


def test_case_model_forbids_extras_and_bad_ids() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        PipelineCase.model_validate(one_case(surprise=1))
    with pytest.raises(ValueError, match="String should match pattern"):
        PipelineCase.model_validate(one_case(case_id="Bad Id"))
