"""Tests for page-aware PDF extraction control flow."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from turbofan_copilot.ingestion.pdf_extraction import extract_pdf_pages
from turbofan_copilot.ingestion.source_manifest import TechnicalDocumentManifest


def chapter_manifest() -> TechnicalDocumentManifest:
    """Return valid planned metadata for an FAA chapter test file."""
    return TechnicalDocumentManifest.model_validate(
        {
            "source_id": "faa-h-8083-32b-chapter-6",
            "kind": "technical_document",
            "title": "Aviation Maintenance Technician Handbook - Powerplant",
            "publisher": "Federal Aviation Administration",
            "version": "FAA-H-8083-32B (2023)",
            "chapter_number": 6,
            "chapter_title": "Lubrication and Cooling Systems",
            "source_url": "https://www.faa.gov/chapter-6",
            "download_url": "https://www.faa.gov/files/chapter-6.pdf",
            "expected_filename": "chapter-6.pdf",
            "license_note": "Retain FAA attribution.",
        }
    )


def test_extract_pages_preserves_order_provenance_and_empty_text(tmp_path: Path) -> None:
    fake_pages = [Mock(), Mock(), Mock()]
    fake_pages[0].extract_text.return_value = "6-1\nFirst page"
    fake_pages[1].extract_text.return_value = "Heading without a printed page label"
    fake_pages[2].extract_text.return_value = None
    fake_reader = Mock(pages=fake_pages)

    with patch(
        "turbofan_copilot.ingestion.pdf_extraction.PdfReader",
        return_value=fake_reader,
    ):
        pages = extract_pdf_pages(tmp_path / "chapter-6.pdf", chapter_manifest())

    assert [page.pdf_page_number for page in pages] == [1, 2, 3]
    assert [page.printed_page_label for page in pages] == ["6-1", None, None]
    assert [page.source_id for page in pages] == [
        "faa-h-8083-32b-chapter-6",
        "faa-h-8083-32b-chapter-6",
        "faa-h-8083-32b-chapter-6",
    ]
    assert pages[2].text == ""


def test_extract_pages_rejects_pdf_filename_that_does_not_match_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Expected PDF filename 'chapter-6.pdf'"):
        extract_pdf_pages(tmp_path / "wrong.pdf", chapter_manifest())
