"""Deterministic reciprocal-rank-fusion hybrid of the lexical and semantic baselines."""

from collections.abc import Iterable

from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.evaluation.retrieval_metrics import evaluate_retriever
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.semantic_retrieval import SupportsRank

type ScoredChunk = tuple[float, DocumentChunk]

RRF_K = 60


def _reciprocal_rank_scores(ranking: tuple[ScoredChunk, ...]) -> dict[DocumentChunk, float]:
    """Map each ranked chunk to its 1 / (k + rank) contribution."""
    return {chunk: 1.0 / (RRF_K + rank) for rank, (_, chunk) in enumerate(ranking, start=1)}


class HybridRetriever:
    """Fuse the full lexical and semantic rankings by reciprocal rank."""

    def __init__(
        self,
        lexical: LexicalRetriever,
        semantic: SupportsRank,
        *,
        corpus_size: int,
    ) -> None:
        if corpus_size <= 0:
            raise ValueError("corpus_size must be greater than zero")
        self._lexical = lexical
        self._semantic = semantic
        self._corpus_size = corpus_size

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        """Return the highest fused-score chunks in a fully deterministic order."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        lexical_scores = _reciprocal_rank_scores(
            self._lexical.rank(question, limit=self._corpus_size)
        )
        semantic_scores = _reciprocal_rank_scores(
            self._semantic.rank(question, limit=self._corpus_size)
        )

        fused = [
            (lexical_scores.get(chunk, 0.0) + semantic_scores.get(chunk, 0.0), chunk)
            for chunk in lexical_scores.keys() | semantic_scores.keys()
        ]
        fused.sort(
            key=lambda item: (
                -item[0],
                item[1].source_id,
                item[1].pdf_page_number,
                item[1].chunk_index,
            )
        )
        return tuple(fused[:limit])


def evaluate_hybrid_retriever(
    cases: Iterable[RetrievalEvaluationCase],
    retriever: HybridRetriever,
) -> dict[str, float]:
    """Return macro-average page metrics for the hybrid retriever."""
    return evaluate_retriever(cases, retriever)
