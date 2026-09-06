"""Deterministic dependency-free lexical retrieval baseline."""

import re
from collections import Counter
from collections.abc import Iterable
from math import log
from statistics import fmean

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.evaluation.retrieval_metrics import (
    page_hit_at_k,
    page_reciprocal_rank,
)
from turbofan_copilot.ingestion.chunking import DocumentChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

type ScoredChunk = tuple[float, DocumentChunk]


def _tokenize(text: str) -> frozenset[str]:
    """Return unique case-insensitive ASCII word and number tokens."""
    return frozenset(TOKEN_PATTERN.findall(text.casefold()))


class LexicalRetriever:
    """Rank chunks by IDF-weighted unique query-term overlap."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = tuple(chunks)
        if not self._chunks:
            raise ValueError("lexical corpus must contain at least one chunk")

        self._chunk_terms = tuple(_tokenize(chunk.text) for chunk in self._chunks)
        self._document_frequencies = Counter(term for terms in self._chunk_terms for term in terms)

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        """Return the highest-scoring positive-overlap chunks."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_terms = _tokenize(question)
        chunk_count = len(self._chunks)
        scored_chunks: list[ScoredChunk] = []
        for chunk, chunk_terms in zip(self._chunks, self._chunk_terms, strict=True):
            score = sum(
                log((chunk_count + 1) / (self._document_frequencies[term] + 1))
                for term in query_terms & chunk_terms
            )
            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                item[1].source_id,
                item[1].pdf_page_number,
                item[1].chunk_index,
            )
        )
        return tuple(scored_chunks[:limit])


def evaluate_lexical_retriever(
    cases: Iterable[RetrievalEvaluationCase],
    retriever: LexicalRetriever,
) -> dict[str, float]:
    """Return macro-average page metrics for the lexical retriever."""
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("lexical evaluation must contain at least one case")

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
