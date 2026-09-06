"""Tests for the normalized extracted-page contract."""

import pytest
from pydantic import ValidationError

from turbofan_copilot.ingestion.document_page import ExtractedPage


def test_page_preserves_text_and_citation_identity() -> None:
    page = ExtractedPage(
        source_id="faa-h-8083-32b-chapter-6",
        chapter_number=6,
        pdf_page_number=14,
        printed_page_label="6-14",
        text="Oil Cooler\nThe oil cooler transfers heat.",
    )

    assert page.printed_page_label == "6-14"
    assert page.text == "Oil Cooler\nThe oil cooler transfers heat."


def test_page_allows_empty_text_for_visible_extraction_failures() -> None:
    page = ExtractedPage(
        source_id="faa-h-8083-32b-chapter-6",
        chapter_number=6,
        pdf_page_number=2,
        text="",
    )

    assert page.text == ""


def test_page_rejects_zero_based_pdf_page_number() -> None:
    with pytest.raises(ValidationError, match="pdf_page_number"):
        ExtractedPage(
            source_id="faa-h-8083-32b-chapter-6",
            chapter_number=6,
            pdf_page_number=0,
            text="Extracted text",
        )
