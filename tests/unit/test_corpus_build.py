"""Tests for deterministic technical-document corpus composition."""

import json
from pathlib import Path
from unittest.mock import patch

from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.ingestion.corpus_build import (
    build_document_corpus,
    load_document_chunks,
)
from turbofan_copilot.ingestion.document_page import ExtractedPage
from turbofan_copilot.ingestion.source_manifest import TechnicalDocumentManifest


def chapter_manifest(chapter_number: int) -> TechnicalDocumentManifest:
    """Return valid metadata for one small corpus-build test document."""
    return TechnicalDocumentManifest.model_validate(
        {
            "source_id": f"faa-chapter-{chapter_number}",
            "kind": "technical_document",
            "title": "FAA Powerplant Handbook",
            "publisher": "Federal Aviation Administration",
            "version": "2023",
            "chapter_number": chapter_number,
            "chapter_title": f"Chapter {chapter_number}",
            "source_url": f"https://www.faa.gov/chapter-{chapter_number}",
            "download_url": f"https://www.faa.gov/chapter-{chapter_number}.pdf",
            "expected_filename": f"chapter-{chapter_number}.pdf",
            "license_note": "Retain FAA attribution.",
        }
    )


def test_build_corpus_composes_stages_and_replaces_output_deterministically(
    tmp_path: Path,
) -> None:
    manifests = [chapter_manifest(6), chapter_manifest(1)]
    manifest_paths = []
    for manifest in manifests:
        path = tmp_path / f"chapter-{manifest.chapter_number}.json"
        path.write_text(manifest.model_dump_json(), encoding="utf-8")
        manifest_paths.append(path)

    def fake_extract_pages(
        path: str | Path,
        manifest: TechnicalDocumentManifest,
    ) -> tuple[ExtractedPage, ...]:
        assert Path(path).name == manifest.expected_filename
        label = f"{manifest.chapter_number}-1"
        return (
            ExtractedPage(
                source_id=manifest.source_id,
                chapter_number=manifest.chapter_number,
                pdf_page_number=1,
                printed_page_label=label,
                text=f"{label}\r\nChapter {manifest.chapter_number} text  \r\n",
            ),
        )

    output_path = tmp_path / "processed" / "manual_chunks.jsonl"
    output_path.parent.mkdir()
    output_path.write_text("stale content", encoding="utf-8")

    with patch(
        "turbofan_copilot.ingestion.corpus_build.extract_pdf_pages",
        side_effect=fake_extract_pages,
    ):
        chunks = build_document_corpus(manifest_paths, tmp_path, output_path)
        first_build = output_path.read_bytes()
        build_document_corpus(reversed(manifest_paths), tmp_path, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [chunk.chapter_number for chunk in chunks] == [1, 6]
    assert [record["text"] for record in records] == [
        "Chapter 1 text\n",
        "Chapter 6 text\n",
    ]
    assert output_path.read_bytes() == first_build
    assert first_build.endswith(b"\n")
    assert b"\r\n" not in first_build
    assert b"stale content" not in first_build


def test_build_corpus_forwards_smaller_chunk_window(tmp_path: Path) -> None:
    manifest = chapter_manifest(1)
    manifest_path = tmp_path / "chapter-1.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    page_text = "".join(f"line {index:02d} of the page\n" for index in range(1, 21))

    def fake_extract_pages(
        path: str | Path,
        manifest: TechnicalDocumentManifest,
    ) -> tuple[ExtractedPage, ...]:
        return (
            ExtractedPage(
                source_id=manifest.source_id,
                chapter_number=manifest.chapter_number,
                pdf_page_number=1,
                printed_page_label="1-1",
                text=page_text,
            ),
        )

    output_path = tmp_path / "small_window.jsonl"
    with patch(
        "turbofan_copilot.ingestion.corpus_build.extract_pdf_pages",
        side_effect=fake_extract_pages,
    ):
        default_chunks = build_document_corpus(
            [manifest_path], tmp_path, tmp_path / "default.jsonl"
        )
        small_chunks = build_document_corpus(
            [manifest_path],
            tmp_path,
            output_path,
            max_characters=60,
            overlap_characters=20,
        )
        first_bytes = output_path.read_bytes()
        build_document_corpus(
            [manifest_path],
            tmp_path,
            output_path,
            max_characters=60,
            overlap_characters=20,
        )

    assert len(default_chunks) == 1
    assert len(small_chunks) > 1
    assert all(len(chunk.text) <= 60 for chunk in small_chunks)
    assert output_path.read_bytes() == first_bytes


def test_load_document_chunks_round_trips_written_corpus_in_order(
    tmp_path: Path,
) -> None:
    chunks = (
        DocumentChunk(
            source_id="faa-h-8083-32b-chapter-1",
            chapter_number=1,
            pdf_page_number=38,
            printed_page_label="1-38",
            chunk_index=2,
            text="centrifugal-flow compressors accelerate air outward",
        ),
        DocumentChunk(
            source_id="faa-h-8083-32b-chapter-6",
            chapter_number=6,
            pdf_page_number=28,
            printed_page_label=None,
            chunk_index=1,
            text="the magnetic chip detector catches ferrous particles",
        ),
    )
    corpus_path = tmp_path / "manual_chunks.jsonl"
    corpus_path.write_text(
        "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks) + "\n",
        encoding="utf-8",
    )

    loaded = load_document_chunks(corpus_path)

    assert loaded == chunks
