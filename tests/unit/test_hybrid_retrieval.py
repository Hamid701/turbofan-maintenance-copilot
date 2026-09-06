"""Tests for the deterministic reciprocal-rank-fusion hybrid retriever."""

from typing import cast

import pytest

from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder, EmbeddingVector
from turbofan_copilot.retrieval.hybrid_retrieval import (
    HybridRetriever,
    evaluate_hybrid_retriever,
)
from turbofan_copilot.retrieval.semantic_retrieval import SemanticRetriever


class FakeQuestionEmbedder:
    """Return one fixed vector for any question without loading a model."""

    def __init__(self, vector: EmbeddingVector) -> None:
        self._vector = vector

    def embed_question(self, question: str) -> EmbeddingVector:
        return self._vector


def as_embedder(fake: FakeQuestionEmbedder) -> BgeEmbedder:
    """Present the fake through the type the retriever expects."""
    return cast(BgeEmbedder, fake)


def chunk(
    text: str,
    pdf_page_number: int,
    *,
    chunk_index: int = 1,
) -> DocumentChunk:
    """Return one Chapter 1 chunk for hybrid tests."""
    return DocumentChunk(
        source_id="faa-h-8083-32b-chapter-1",
        chapter_number=1,
        pdf_page_number=pdf_page_number,
        printed_page_label=None,
        chunk_index=chunk_index,
        text=text,
    )


def evaluation_case(
    case_id: str,
    question: str,
    expected_pdf_page: int,
) -> RetrievalEvaluationCase:
    """Return one Chapter 1 page-grounded case."""
    return RetrievalEvaluationCase(
        case_id=case_id,
        question=question,
        expected_source_id="faa-h-8083-32b-chapter-1",
        expected_pdf_pages=(expected_pdf_page,),
        relevance_note="Verified test evidence.",
    )


def test_fusion_rewards_chunks_ranked_high_in_both_lists() -> None:
    both = chunk("alpha beta", 1)
    lexical_only = chunk("alpha", 2)
    semantic_only = chunk("gamma", 3)
    chunks = [both, lexical_only, semantic_only]

    lexical = LexicalRetriever(chunks)
    semantic = SemanticRetriever(
        chunks,
        [(1.0, 0.0), (0.6, 0.8), (0.0, 1.0)],
        as_embedder(FakeQuestionEmbedder((1.0, 0.05))),
    )
    hybrid = HybridRetriever(lexical, semantic, corpus_size=len(chunks))

    ordered_pages = [ranked.pdf_page_number for _, ranked in hybrid.rank("alpha beta", limit=3)]

    assert ordered_pages == [1, 2, 3]


def test_ranking_is_deterministic_and_total_ordered_under_tied_fused_scores() -> None:
    early = chunk("alpha", 2)
    late = chunk("alpha", 5)
    other = chunk("beta", 9)
    chunks = [early, late, other]

    # Lexical ties early and late (equal IDF), so its tie-break orders them early, late.
    # Semantic ranks late first, then early, then other. The two reciprocal-rank
    # contributions for early and late are the same pair of floats added in each order,
    # so their fused scores are exactly equal and the metadata tie-break decides.
    lexical = LexicalRetriever(chunks)
    semantic = SemanticRetriever(
        chunks,
        [(0.9, 0.1), (1.0, 0.0), (0.0, 1.0)],
        as_embedder(FakeQuestionEmbedder((1.0, 0.0))),
    )
    hybrid = HybridRetriever(lexical, semantic, corpus_size=len(chunks))

    first = [ranked.pdf_page_number for _, ranked in hybrid.rank("alpha", limit=3)]
    second = [ranked.pdf_page_number for _, ranked in hybrid.rank("alpha", limit=3)]

    assert first == [2, 5, 9]
    assert first == second


def test_hybrid_rejects_nonpositive_corpus_size_and_limit() -> None:
    chunks = [chunk("alpha", 1)]
    lexical = LexicalRetriever(chunks)
    semantic = SemanticRetriever(
        chunks, [(1.0, 0.0)], as_embedder(FakeQuestionEmbedder((1.0, 0.0)))
    )

    with pytest.raises(ValueError, match="corpus_size must be greater than zero"):
        HybridRetriever(lexical, semantic, corpus_size=0)

    hybrid = HybridRetriever(lexical, semantic, corpus_size=1)
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        hybrid.rank("alpha", limit=0)


def test_evaluate_macro_averages_page_metrics() -> None:
    page_one = chunk("alpha evidence", 1)
    page_two = chunk("beta evidence", 2)
    chunks = [page_one, page_two]

    lexical = LexicalRetriever(chunks)
    semantic = SemanticRetriever(
        chunks,
        [(1.0, 0.0), (0.0, 1.0)],
        as_embedder(FakeQuestionEmbedder((1.0, 0.0))),
    )
    hybrid = HybridRetriever(lexical, semantic, corpus_size=len(chunks))
    cases = [
        evaluation_case("alpha_case", "alpha", 1),
        evaluation_case("beta_case", "beta", 2),
    ]

    metrics = evaluate_hybrid_retriever(cases, hybrid)

    assert metrics == {
        "hit_at_1": 1.0,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mean_reciprocal_rank": 1.0,
    }


def test_evaluate_rejects_empty_case_collection() -> None:
    chunks = [chunk("alpha", 1)]
    lexical = LexicalRetriever(chunks)
    semantic = SemanticRetriever(
        chunks, [(1.0, 0.0)], as_embedder(FakeQuestionEmbedder((1.0, 0.0)))
    )
    hybrid = HybridRetriever(lexical, semantic, corpus_size=1)

    with pytest.raises(ValueError, match="at least one case"):
        evaluate_hybrid_retriever([], hybrid)
