"""Tests for raw-file presence validation."""

from pathlib import Path

import pytest

from turbofan_copilot.ingestion.raw_files import validate_expected_files
from turbofan_copilot.ingestion.source_manifest import SourceManifest


def manifest_for(*expected_files: str) -> SourceManifest:
    """Build a valid manifest containing the requested filenames."""
    return SourceManifest.model_validate(
        {
            "source_id": "test-source",
            "kind": "dataset",
            "title": "Test source",
            "publisher": "Test publisher",
            "source_url": "https://example.com/source",
            "download_url": "https://example.com/source.zip",
            "version": "test-version",
            "license_note": "Test-only metadata.",
            "expected_files": list(expected_files),
        }
    )


def test_validate_expected_files_returns_paths_in_manifest_order(tmp_path: Path) -> None:
    manifest = manifest_for("train.txt", "test.txt", "rul.txt")
    for name in manifest.expected_files:
        (tmp_path / name).write_text("test data", encoding="utf-8")

    paths = validate_expected_files(manifest, tmp_path)

    assert paths == (
        tmp_path / "train.txt",
        tmp_path / "test.txt",
        tmp_path / "rul.txt",
    )


def test_validate_expected_files_reports_all_missing_names(tmp_path: Path) -> None:
    manifest = manifest_for("present.txt", "missing-a.txt", "missing-b.txt")
    (tmp_path / "present.txt").write_text("test data", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        validate_expected_files(manifest, tmp_path)

    message = str(error.value)
    assert "missing-a.txt" in message
    assert "missing-b.txt" in message
    assert "present.txt" not in message
