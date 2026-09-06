"""Tests for the deterministic lexical retrieval baseline."""

import pytest

from turbofan_copilot.evaluation.lexical_retrieval import (
    LexicalRetriever,
    evaluate_lexical_retriever,
)
from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.ingestion.chunking import DocumentChunk


def corpus_chunk(
    text: str,
    pdf_page_number: int,
    *,
    chunk_index: int = 1,
) -> DocumentChunk:
    """Return one Chapter 1 chunk for lexical tests."""
    return DocumentChunk(
        source_id="faa-h-8083-32b-chapter-1",
        chapter_number=1,
        pdf_page_number=pdf_page_number,
        printed_page_label=f"1-{pdf_page_number}",
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


def test_rank_normalizes_text_weights_rare_terms_and_excludes_zero_scores() -> None:
    retriever = LexicalRetriever(
        [
            corpus_chunk("engine Brayton cycle", 58),
            corpus_chunk("engine accessories", 38),
            corpus_chunk("engine", 1),
        ]
    )

    results = retriever.rank("BRAYTON-cycle engine!", limit=5)

    assert [(chunk.pdf_page_number, score > 0) for score, chunk in results] == [(58, True)]


def test_rank_uses_metadata_ties_and_respects_limit() -> None:
    retriever = LexicalRetriever(
        [
            corpus_chunk("magnetic particles", 28),
            corpus_chunk("magnetic field", 20),
            corpus_chunk("unrelated evidence", 1),
        ]
    )

    results = retriever.rank("magnetic", limit=1)

    assert [chunk.pdf_page_number for _, chunk in results] == [20]


def test_retriever_rejects_empty_corpus_and_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="corpus must contain at least one chunk"):
        LexicalRetriever([])

    retriever = LexicalRetriever([corpus_chunk("alpha", 1), corpus_chunk("beta", 2)])
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        retriever.rank("alpha", limit=0)


def test_evaluation_macro_averages_hits_and_empty_results() -> None:
    retriever = LexicalRetriever(
        [corpus_chunk("alpha evidence", 1), corpus_chunk("beta evidence", 2)]
    )
    cases = [
        evaluation_case("alpha_case", "alpha", 1),
        evaluation_case("missing_case", "missing", 2),
    ]

    metrics = evaluate_lexical_retriever(cases, retriever)

    assert metrics == {
        "hit_at_1": 0.5,
        "hit_at_3": 0.5,
        "hit_at_5": 0.5,
        "mean_reciprocal_rank": 0.5,
    }


def test_evaluation_rejects_empty_case_collection() -> None:
    retriever = LexicalRetriever([corpus_chunk("alpha", 1), corpus_chunk("beta", 2)])

    with pytest.raises(ValueError, match="evaluation must contain at least one case"):
        evaluate_lexical_retriever([], retriever)
