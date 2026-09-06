"""Tests for page-grounded retrieval evaluation contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from turbofan_copilot.evaluation.retrieval_case import (
    RetrievalEvaluationCase,
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.ingestion.chunking import DocumentChunk


def valid_case_data() -> dict[str, object]:
    """Return one valid retrieval evaluation case."""
    return {
        "case_id": "oil_pressure_causes",
        "question": " What conditions can cause low engine oil pressure? ",
        "expected_source_id": "faa-h-8083-32b-chapter-6",
        "expected_pdf_pages": [12, 13],
        "relevance_note": " These pages explain relevant oil-pressure conditions. ",
    }


def corpus_chunk(pdf_page_number: int) -> DocumentChunk:
    """Return one valid corpus chunk for an expected Chapter 6 page."""
    return DocumentChunk(
        source_id="faa-h-8083-32b-chapter-6",
        chapter_number=6,
        pdf_page_number=pdf_page_number,
        printed_page_label=f"6-{pdf_page_number}",
        chunk_index=1,
        text="Relevant technical evidence.",
    )


def test_loader_returns_trimmed_frozen_cases(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_cases.json"
    path.write_text(json.dumps([valid_case_data()]), encoding="utf-8")

    cases = load_retrieval_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].question == "What conditions can cause low engine oil pressure?"
    assert cases[0].expected_pdf_pages == (12, 13)
    assert cases[0].relevance_note == "These pages explain relevant oil-pressure conditions."


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        ([13, 12], "unique and ascending"),
        ([12, 12], "unique and ascending"),
        ([0], "greater than 0"),
    ],
)
def test_case_rejects_invalid_expected_pages(
    pages: list[int],
    message: str,
) -> None:
    data = valid_case_data()
    data["expected_pdf_pages"] = pages

    with pytest.raises(ValidationError, match=message):
        RetrievalEvaluationCase.model_validate(data)


def test_case_rejects_unknown_fields() -> None:
    data = valid_case_data()
    data["keywords"] = ["oil", "pressure"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrievalEvaluationCase.model_validate(data)


@pytest.mark.parametrize(
    ("raw_data", "message"),
    [
        ({}, "must contain a JSON array"),
        ([], "must contain at least one case"),
        ([valid_case_data(), valid_case_data()], "case_id values must be unique"),
    ],
)
def test_loader_rejects_invalid_case_collections(
    tmp_path: Path,
    raw_data: object,
    message: str,
) -> None:
    path = tmp_path / "retrieval_cases.json"
    path.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_retrieval_evaluation_cases(path)


def test_page_validation_accepts_expected_pages_present_in_corpus() -> None:
    case = RetrievalEvaluationCase.model_validate(valid_case_data())

    validate_retrieval_case_pages([case], [corpus_chunk(12), corpus_chunk(13)])


def test_page_validation_reports_missing_case_and_page() -> None:
    case = RetrievalEvaluationCase.model_validate(valid_case_data())

    with pytest.raises(
        ValueError,
        match=r"oil_pressure_causes .* page 13",
    ):
        validate_retrieval_case_pages([case], [corpus_chunk(12)])
