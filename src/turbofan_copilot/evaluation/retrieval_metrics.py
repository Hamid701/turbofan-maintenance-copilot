"""Pure page-grounded retrieval metrics."""

from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Protocol

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.ingestion.chunking import DocumentChunk

DEFAULT_BUDGET = 5


class RanksChunks(Protocol):
    """Structural shape of any retriever, declared here to avoid an import cycle.

    Matches ``turbofan_copilot.retrieval.semantic_retrieval.SupportsRank``.
    """

    def rank(
        self,
        question: str,
        *,
        limit: int = ...,
    ) -> tuple[tuple[float, DocumentChunk], ...]: ...


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


def evaluate_retriever(
    cases: Iterable[RetrievalEvaluationCase],
    retriever: RanksChunks,
    *,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, float]:
    """Return macro-average page metrics for any retriever.

    This is the single implementation; the lexical, semantic, and hybrid modules
    all delegate here so the three baselines cannot drift apart.
    """
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("retrieval evaluation must contain at least one case")

    hits_at_1: list[int] = []
    hits_at_3: list[int] = []
    hits_at_5: list[int] = []
    reciprocal_ranks: list[float] = []
    for case in case_list:
        ranked = tuple(chunk for _, chunk in retriever.rank(case.question, limit=budget))
        hits_at_1.append(page_hit_at_k(case, ranked, k=1))
        hits_at_3.append(page_hit_at_k(case, ranked, k=3))
        hits_at_5.append(page_hit_at_k(case, ranked, k=5))
        reciprocal_ranks.append(page_reciprocal_rank(case, ranked))

    return {
        "hit_at_1": fmean(hits_at_1),
        "hit_at_3": fmean(hits_at_3),
        "hit_at_5": fmean(hits_at_5),
        "mean_reciprocal_rank": fmean(reciprocal_ranks),
    }
