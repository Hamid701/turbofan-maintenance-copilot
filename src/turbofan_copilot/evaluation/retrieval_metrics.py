"""Pure page-grounded retrieval metrics."""

from collections.abc import Sequence

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.ingestion.chunking import DocumentChunk


def _is_relevant_page(
    case: RetrievalEvaluationCase,
    chunk: DocumentChunk,
) -> bool:
    """Return whether one chunk belongs to an expected source page."""
    return (
        chunk.source_id == case.expected_source_id
        and chunk.pdf_page_number in case.expected_pdf_pages
    )


def page_hit_at_k(
    case: RetrievalEvaluationCase,
    ranked_chunks: Sequence[DocumentChunk],
    k: int,
) -> int:
    """Return one when an expected page occurs in the first k chunks."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return int(any(_is_relevant_page(case, chunk) for chunk in ranked_chunks[:k]))


def page_reciprocal_rank(
    case: RetrievalEvaluationCase,
    ranked_chunks: Sequence[DocumentChunk],
) -> float:
    """Return the inverse rank of the first chunk from an expected page."""
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if _is_relevant_page(case, chunk):
            return 1.0 / rank
    return 0.0
