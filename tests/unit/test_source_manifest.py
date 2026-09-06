"""Tests for the external-source provenance contract."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from turbofan_copilot.ingestion.source_manifest import (
    SourceKind,
    SourceManifest,
    load_source_manifest,
)


def valid_manifest_data() -> dict[str, object]:
    """Return raw data shaped like a manifest read from JSON."""
    return {
        "source_id": "nasa-cmapss-fd001",
        "kind": "dataset",
        "title": "C-MAPSS FD001",
        "publisher": "NASA",
        "source_url": "https://example.nasa.gov/cmapss",
        "download_url": "https://example.nasa.gov/files/cmapss.zip",
        "version": "FD001",
        "license_note": "Usage terms pending review.",
        "expected_files": ["train_FD001.txt"],
    }


def test_manifest_accepts_complete_retrieval_evidence() -> None:
    manifest = SourceManifest.model_validate(
        {
            "source_id": "nasa-cmapss-fd001",
            "kind": "dataset",
            "title": "C-MAPSS FD001",
            "publisher": "NASA",
            "source_url": "https://example.nasa.gov/cmapss.zip",
            "download_url": "https://example.nasa.gov/files/cmapss.zip",
            "version": "FD001",
            "license_note": "Record the publisher's usage terms before distribution.",
            "expected_files": ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"],
            "retrieved_on": date(2026, 8, 1),
            "archive_sha256": "a" * 64,
        }
    )

    assert manifest.kind is SourceKind.DATASET
    assert len(manifest.expected_files) == 3


def test_manifest_rejects_incomplete_retrieval_evidence() -> None:
    with pytest.raises(ValidationError, match="must be provided together"):
        SourceManifest.model_validate(
            {
                "source_id": "nasa-cmapss-fd001",
                "kind": "dataset",
                "title": "C-MAPSS FD001",
                "publisher": "NASA",
                "source_url": "https://example.nasa.gov/cmapss.zip",
                "download_url": "https://example.nasa.gov/files/cmapss.zip",
                "version": "FD001",
                "license_note": "Usage terms pending review.",
                "expected_files": ["train_FD001.txt"],
                "retrieved_on": date(2026, 8, 1),
            }
        )


def test_load_source_manifest_returns_validated_model(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(valid_manifest_data()), encoding="utf-8")

    manifest = load_source_manifest(manifest_path)

    assert manifest.source_id == "nasa-cmapss-fd001"
    assert manifest.kind is SourceKind.DATASET


def test_load_source_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"source_id":', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_source_manifest(manifest_path)


def test_load_source_manifest_rejects_invalid_fields(tmp_path: Path) -> None:
    manifest_data = valid_manifest_data()
    manifest_data["source_id"] = "NASA FD001"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValidationError, match="source_id"):
        load_source_manifest(manifest_path)
