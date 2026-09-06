"""Tests for technical-document provenance metadata."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from turbofan_copilot.ingestion.source_manifest import (
    SourceKind,
    TechnicalDocumentManifest,
    load_technical_document_manifest,
)


def valid_document_data() -> dict[str, object]:
    """Return planned metadata for one FAA handbook chapter."""
    return {
        "source_id": "faa-h-8083-32b-chapter-1",
        "kind": "technical_document",
        "title": "Aviation Maintenance Technician Handbook - Powerplant",
        "publisher": "Federal Aviation Administration",
        "version": "FAA-H-8083-32B (2023)",
        "chapter_number": 1,
        "chapter_title": "Aircraft Engines",
        "source_url": "https://www.faa.gov/chapter-1",
        "download_url": "https://www.faa.gov/files/03_amtp_ch1.pdf",
        "expected_filename": "03_amtp_ch1.pdf",
        "license_note": "Retain FAA attribution and verify redistribution terms.",
    }


def test_manifest_accepts_planned_document_metadata() -> None:
    manifest = TechnicalDocumentManifest.model_validate(valid_document_data())

    assert manifest.kind is SourceKind.TECHNICAL_DOCUMENT
    assert manifest.chapter_number == 1
    assert manifest.retrieved_on is None


def test_loader_accepts_complete_file_evidence(tmp_path: Path) -> None:
    data = valid_document_data()
    data["retrieved_on"] = "2026-08-01"
    data["file_sha256"] = "a" * 64
    path = tmp_path / "chapter-1.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = load_technical_document_manifest(path)

    assert manifest.retrieved_on == date(2026, 8, 1)
    assert manifest.file_sha256 == "a" * 64


def test_manifest_rejects_incomplete_file_evidence() -> None:
    data = valid_document_data()
    data["retrieved_on"] = "2026-08-01"

    with pytest.raises(ValidationError, match="must be provided together"):
        TechnicalDocumentManifest.model_validate(data)


def test_manifest_rejects_path_instead_of_pdf_filename() -> None:
    data = valid_document_data()
    data["expected_filename"] = "../03_amtp_ch1.pdf"

    with pytest.raises(ValidationError, match="expected_filename"):
        TechnicalDocumentManifest.model_validate(data)
