"""Tests for pure page-grounded retrieval metrics."""

import pytest

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.evaluation.retrieval_metrics import (
    page_hit_at_k,
    page_reciprocal_rank,
)
from turbofan_copilot.ingestion.chunking import DocumentChunk


def evaluation_case() -> RetrievalEvaluationCase:
    """Return a case expecting Chapter 6, PDF page 28."""
    return RetrievalEvaluationCase(
        case_id="magnetic_chip_detector_particles",
        question="How does a magnetic chip detector identify possible engine wear?",
        expected_source_id="faa-h-8083-32b-chapter-6",
        expected_pdf_pages=(28,),
        relevance_note="Page 6-28 explains magnetic chip detectors.",
    )


def corpus_chunk(
    source_id: str,
    pdf_page_number: int,
    chunk_index: int = 1,
) -> DocumentChunk:
    """Return one ranked corpus chunk with stable page provenance."""
    return DocumentChunk(
        source_id=source_id,
        chapter_number=6,
        pdf_page_number=pdf_page_number,
        printed_page_label=f"6-{pdf_page_number}",
        chunk_index=chunk_index,
        text="Technical evidence.",
    )


def test_page_hit_at_k_respects_the_cutoff() -> None:
    ranking = [
        corpus_chunk("faa-h-8083-32b-chapter-6", 20),
        corpus_chunk("faa-h-8083-32b-chapter-6", 28),
    ]

    assert page_hit_at_k(evaluation_case(), ranking, k=1) == 0
    assert page_hit_at_k(evaluation_case(), ranking, k=2) == 1


@pytest.mark.parametrize("k", [0, -1])
def test_page_hit_at_k_rejects_nonpositive_cutoffs(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        page_hit_at_k(evaluation_case(), [], k=k)


def test_page_reciprocal_rank_uses_the_first_exact_page_match() -> None:
    ranking = [
        corpus_chunk("faa-h-8083-32b-chapter-1", 28),
        corpus_chunk("faa-h-8083-32b-chapter-6", 28),
        corpus_chunk("faa-h-8083-32b-chapter-6", 28),
    ]

    assert page_reciprocal_rank(evaluation_case(), ranking) == 0.5


def test_empty_ranking_scores_zero() -> None:
    assert page_hit_at_k(evaluation_case(), [], k=5) == 0
    assert page_reciprocal_rank(evaluation_case(), []) == 0.0


@pytest.mark.parametrize(
    ("ranking", "expected_hits", "expected_reciprocal_rank"),
    [
        (
            [corpus_chunk("faa-h-8083-32b-chapter-6", 28)],
            (1, 1, 1),
            1.0,
        ),
        (
            [
                corpus_chunk("faa-h-8083-32b-chapter-6", 20),
                corpus_chunk("faa-h-8083-32b-chapter-6", 21),
                corpus_chunk("faa-h-8083-32b-chapter-6", 28),
            ],
            (0, 1, 1),
            1 / 3,
        ),
        (
            [
                corpus_chunk("faa-h-8083-32b-chapter-6", 20, chunk_index=1),
                corpus_chunk("faa-h-8083-32b-chapter-6", 20, chunk_index=2),
                corpus_chunk("faa-h-8083-32b-chapter-6", 28),
            ],
            (0, 1, 1),
            1 / 3,
        ),
        (
            [
                corpus_chunk("faa-h-8083-32b-chapter-6", 20),
                corpus_chunk("faa-h-8083-32b-chapter-6", 21),
            ],
            (0, 0, 0),
            0.0,
        ),
    ],
)
def test_metrics_match_hand_worked_rankings(
    ranking: list[DocumentChunk],
    expected_hits: tuple[int, int, int],
    expected_reciprocal_rank: float,
) -> None:
    case = evaluation_case()

    hits = tuple(page_hit_at_k(case, ranking, k=k) for k in (1, 3, 5))

    assert hits == expected_hits
    assert page_reciprocal_rank(case, ranking) == pytest.approx(expected_reciprocal_rank)
