"""Page-aware text extraction from technical PDF documents."""

from pathlib import Path

from pypdf import PdfReader

from turbofan_copilot.ingestion.document_page import ExtractedPage
from turbofan_copilot.ingestion.source_manifest import TechnicalDocumentManifest


def _printed_page_label(
    text: str,
    chapter_number: int,
    pdf_page_number: int,
) -> str | None:
    """Return the expected FAA page label when the extracted text confirms it."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), None)
    expected_label = f"{chapter_number}-{pdf_page_number}"
    return expected_label if first_line == expected_label else None


def extract_pdf_pages(
    path: str | Path,
    manifest: TechnicalDocumentManifest,
) -> tuple[ExtractedPage, ...]:
    """Extract every physical PDF page with its citation identity."""
    pdf_path = Path(path)
    if pdf_path.name != manifest.expected_filename:
        raise ValueError(
            f"Expected PDF filename {manifest.expected_filename!r}, received {pdf_path.name!r}"
        )

    reader = PdfReader(pdf_path)
    extracted_pages = []
    for pdf_page_number, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        extracted_pages.append(
            ExtractedPage(
                source_id=manifest.source_id,
                chapter_number=manifest.chapter_number,
                pdf_page_number=pdf_page_number,
                printed_page_label=_printed_page_label(
                    text,
                    manifest.chapter_number,
                    pdf_page_number,
                ),
                text=text,
            )
        )

    return tuple(extracted_pages)
