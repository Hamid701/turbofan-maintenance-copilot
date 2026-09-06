"""Deterministic in-memory cosine ranking over BGE corpus vectors."""

from collections.abc import Iterable
from math import fsum
from statistics import fmean
from typing import Protocol

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.evaluation.retrieval_metrics import (
    page_hit_at_k,
    page_reciprocal_rank,
)
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder, EmbeddingVector

type ScoredChunk = tuple[float, DocumentChunk]


class SupportsRank(Protocol):
    """Any retriever that returns scored chunks for a question, best first."""

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]: ...


def cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    """Return the dot product of two vectors, which is cosine similarity for unit vectors."""
    return fsum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


class SemanticRetriever:
    """Rank page-contained chunks by cosine similarity to an embedded question."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        vectors: Iterable[EmbeddingVector],
        embedder: BgeEmbedder,
    ) -> None:
        self._chunks = tuple(chunks)
        self._vectors = tuple(vectors)
        if not self._chunks:
            raise ValueError("semantic corpus must contain at least one chunk")
        if len(self._vectors) != len(self._chunks):
            raise ValueError("semantic corpus needs exactly one vector per chunk")
        self._embedder = embedder

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        """Return the highest cosine-similarity chunks in a fully deterministic order."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        question_vector = self._embedder.embed_question(question)
        scored_chunks = [
            (cosine_similarity(question_vector, vector), chunk)
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                item[1].source_id,
                item[1].pdf_page_number,
                item[1].chunk_index,
            )
        )
        return tuple(scored_chunks[:limit])


def evaluate_semantic_retriever(
    cases: Iterable[RetrievalEvaluationCase],
    retriever: SupportsRank,
) -> dict[str, float]:
    """Return macro-average page metrics for any question-to-chunk retriever."""
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("semantic evaluation must contain at least one case")

    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    reciprocal_ranks = []
    for case in case_list:
        ranked_chunks = tuple(chunk for _, chunk in retriever.rank(case.question, limit=5))
        hits_at_1.append(page_hit_at_k(case, ranked_chunks, k=1))
        hits_at_3.append(page_hit_at_k(case, ranked_chunks, k=3))
        hits_at_5.append(page_hit_at_k(case, ranked_chunks, k=5))
        reciprocal_ranks.append(page_reciprocal_rank(case, ranked_chunks))

    return {
        "hit_at_1": fmean(hits_at_1),
        "hit_at_3": fmean(hits_at_3),
        "hit_at_5": fmean(hits_at_5),
        "mean_reciprocal_rank": fmean(reciprocal_ranks),
    }
