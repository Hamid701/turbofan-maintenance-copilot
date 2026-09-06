"""Page-grounded retrieval evaluation cases."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from turbofan_copilot.ingestion.chunking import DocumentChunk


class RetrievalEvaluationCase(BaseModel):
    """One question paired with its verified relevant source pages."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1)
    expected_source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    expected_pdf_pages: tuple[PositiveInt, ...] = Field(min_length=1)
    relevance_note: str = Field(min_length=1)

    @field_validator("expected_pdf_pages")
    @classmethod
    def require_unique_ascending_pages(
        cls,
        pages: tuple[PositiveInt, ...],
    ) -> tuple[PositiveInt, ...]:
        """Reject ambiguous duplicate or unordered page expectations."""
        if tuple(sorted(set(pages))) != pages:
            raise ValueError("expected_pdf_pages must be unique and ascending")
        return pages


def load_retrieval_evaluation_cases(
    path: str | Path,
) -> tuple[RetrievalEvaluationCase, ...]:
    """Load and validate a non-empty JSON array of retrieval cases."""
    evaluation_path = Path(path)
    raw_data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("retrieval evaluation file must contain a JSON array")

    cases = tuple(RetrievalEvaluationCase.model_validate(item) for item in raw_data)
    if not cases:
        raise ValueError("retrieval evaluation file must contain at least one case")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval evaluation case_id values must be unique")

    return cases


def validate_retrieval_case_pages(
    cases: Iterable[RetrievalEvaluationCase],
    chunks: Iterable[DocumentChunk],
) -> None:
    """Require every expected source page to exist in the corpus."""
    available_pages = {(chunk.source_id, chunk.pdf_page_number) for chunk in chunks}
    missing_pages = [
        f"{case.case_id} ({case.expected_source_id}, page {page_number})"
        for case in cases
        for page_number in case.expected_pdf_pages
        if (case.expected_source_id, page_number) not in available_pages
    ]
    if missing_pages:
        missing_details = "; ".join(missing_pages)
        raise ValueError(f"retrieval evaluation pages not found in corpus: {missing_details}")
