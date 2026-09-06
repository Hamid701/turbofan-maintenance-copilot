"""Deterministic composition of the technical-document ingestion stages."""

from collections.abc import Iterable
from pathlib import Path

from turbofan_copilot.ingestion.chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_OVERLAP_CHARACTERS,
    DocumentChunk,
    chunk_page_text,
)
from turbofan_copilot.ingestion.pdf_extraction import extract_pdf_pages
from turbofan_copilot.ingestion.source_manifest import (
    load_technical_document_manifest,
)
from turbofan_copilot.ingestion.text_normalization import normalize_page_text


def build_document_corpus(
    manifest_paths: Iterable[str | Path],
    raw_directory: str | Path,
    output_path: str | Path,
    *,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> tuple[DocumentChunk, ...]:
    """Build an ordered JSONL corpus from technical-document manifests."""
    manifests = sorted(
        (load_technical_document_manifest(path) for path in manifest_paths),
        key=lambda manifest: manifest.chapter_number,
    )
    raw_path = Path(raw_directory)
    chunks: list[DocumentChunk] = []

    for manifest in manifests:
        pages = extract_pdf_pages(raw_path / manifest.expected_filename, manifest)
        for page in pages:
            normalized_text = normalize_page_text(page.text, page.printed_page_label)
            chunks.extend(
                chunk_page_text(
                    page,
                    normalized_text,
                    max_characters=max_characters,
                    overlap_characters=overlap_characters,
                )
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    json_lines = "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks)
    destination.write_text(json_lines, encoding="utf-8", newline="\n")

    return tuple(chunks)


def load_document_chunks(path: str | Path) -> tuple[DocumentChunk, ...]:
    """Load a built JSONL corpus back into validated chunks in file order."""
    source = Path(path)
    return tuple(
        DocumentChunk.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
