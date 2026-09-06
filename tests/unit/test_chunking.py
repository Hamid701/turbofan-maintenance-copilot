"""Tests for the page-contained retrieval chunking baseline."""

import pytest

from turbofan_copilot.ingestion.chunking import chunk_page_text
from turbofan_copilot.ingestion.document_page import ExtractedPage


def extracted_page(text: str = "Raw page text") -> ExtractedPage:
    """Return one page with stable citation metadata."""
    return ExtractedPage(
        source_id="faa-h-8083-32b-chapter-6",
        chapter_number=6,
        pdf_page_number=14,
        printed_page_label="6-14",
        text=text,
    )


def test_short_page_produces_one_chunk_with_page_provenance() -> None:
    chunks = chunk_page_text(extracted_page(), "Oil cooler\nOperating limits")

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 1
    assert chunks[0].pdf_page_number == 14
    assert chunks[0].printed_page_label == "6-14"
    assert chunks[0].text == "Oil cooler\nOperating limits"


def test_multiple_chunks_overlap_by_complete_lines() -> None:
    chunks = chunk_page_text(
        extracted_page(),
        "a1\na2\na3\na4\na5",
        max_characters=9,
        overlap_characters=3,
    )

    assert [chunk.chunk_index for chunk in chunks] == [1, 2]
    assert [chunk.text for chunk in chunks] == ["a1\na2\na3\n", "a3\na4\na5"]


def test_whitespace_page_is_skipped_and_oversized_line_is_preserved() -> None:
    assert chunk_page_text(extracted_page(), " \n\t") == ()

    chunks = chunk_page_text(
        extracted_page(),
        "abcdefghij\nx",
        max_characters=5,
        overlap_characters=1,
    )

    assert [chunk.text for chunk in chunks] == ["abcdefghij\n", "x"]


@pytest.mark.parametrize(
    ("max_characters", "overlap_characters", "message"),
    [
        (0, 0, "max_characters must be greater than zero"),
        (10, -1, "overlap_characters must be nonnegative"),
        (10, 10, "overlap_characters must be smaller than max_characters"),
    ],
)
def test_invalid_size_parameters_are_rejected(
    max_characters: int,
    overlap_characters: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_page_text(
            extracted_page(),
            "text",
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
