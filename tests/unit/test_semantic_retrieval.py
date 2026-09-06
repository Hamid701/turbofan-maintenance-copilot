"""Tests for the deterministic in-memory semantic retrieval baseline."""

from typing import cast

import pytest

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder, EmbeddingVector
from turbofan_copilot.retrieval.semantic_retrieval import (
    SemanticRetriever,
    cosine_similarity,
    evaluate_semantic_retriever,
)


class FakeQuestionEmbedder:
    """Return one fixed vector for any question without loading a model."""

    def __init__(self, vector: EmbeddingVector) -> None:
        self._vector = vector
        self.seen_questions: list[str] = []

    def embed_question(self, question: str) -> EmbeddingVector:
        self.seen_questions.append(question)
        return self._vector


def as_embedder(fake: FakeQuestionEmbedder) -> BgeEmbedder:
    """Present the fake through the type the retriever expects."""
    return cast(BgeEmbedder, fake)


def corpus_chunk(
    pdf_page_number: int,
    *,
    chunk_index: int = 1,
    source_id: str = "faa-h-8083-32b-chapter-1",
) -> DocumentChunk:
    """Return one page-contained chunk for semantic tests."""
    return DocumentChunk(
        source_id=source_id,
        chapter_number=1,
        pdf_page_number=pdf_page_number,
        printed_page_label=None,
        chunk_index=chunk_index,
        text=f"chunk {source_id} {pdf_page_number} {chunk_index}",
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


def test_cosine_similarity_is_a_length_checked_dot_product() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="argument 2 is longer"):
        cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))


def test_rank_orders_by_similarity_then_by_page_metadata_on_ties() -> None:
    chunks = [corpus_chunk(5), corpus_chunk(9), corpus_chunk(2)]
    vectors: tuple[EmbeddingVector, ...] = ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
    retriever = SemanticRetriever(chunks, vectors, as_embedder(FakeQuestionEmbedder((1.0, 0.0))))

    results = retriever.rank("any question", limit=3)

    assert [(round(score, 6), chunk.pdf_page_number) for score, chunk in results] == [
        (1.0, 2),
        (1.0, 5),
        (0.0, 9),
    ]


def test_rank_is_deterministic_across_repeated_calls_under_exact_ties() -> None:
    chunks = [corpus_chunk(7, chunk_index=2), corpus_chunk(7, chunk_index=1)]
    vectors: tuple[EmbeddingVector, ...] = ((1.0, 0.0), (1.0, 0.0))
    retriever = SemanticRetriever(chunks, vectors, as_embedder(FakeQuestionEmbedder((1.0, 0.0))))

    first = [chunk.chunk_index for _, chunk in retriever.rank("q", limit=2)]
    second = [chunk.chunk_index for _, chunk in retriever.rank("q", limit=2)]

    assert first == [1, 2]
    assert first == second


def test_retriever_rejects_bad_shape_and_nonpositive_limit() -> None:
    fake = as_embedder(FakeQuestionEmbedder((1.0, 0.0)))

    with pytest.raises(ValueError, match="at least one chunk"):
        SemanticRetriever([], [], fake)

    with pytest.raises(ValueError, match="one vector per chunk"):
        SemanticRetriever([corpus_chunk(1)], [], fake)

    retriever = SemanticRetriever([corpus_chunk(1)], [(1.0, 0.0)], fake)
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        retriever.rank("q", limit=0)


def test_evaluate_macro_averages_page_metrics() -> None:
    chunks = [corpus_chunk(1), corpus_chunk(2)]
    vectors: tuple[EmbeddingVector, ...] = ((1.0, 0.0), (0.0, 1.0))
    retriever = SemanticRetriever(chunks, vectors, as_embedder(FakeQuestionEmbedder((1.0, 0.0))))
    cases = [
        evaluation_case("page_one_case", "matches page one", 1),
        evaluation_case("page_two_case", "still matches page one", 2),
    ]

    metrics = evaluate_semantic_retriever(cases, retriever)

    assert metrics == {
        "hit_at_1": 0.5,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mean_reciprocal_rank": 0.75,
    }


def test_evaluate_rejects_empty_case_collection() -> None:
    retriever = SemanticRetriever(
        [corpus_chunk(1)], [(1.0, 0.0)], as_embedder(FakeQuestionEmbedder((1.0, 0.0)))
    )

    with pytest.raises(ValueError, match="at least one case"):
        evaluate_semantic_retriever([], retriever)
