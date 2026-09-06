"""Typed provenance records for external project data sources."""

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class SourceKind(StrEnum):
    """Kinds of external evidence used by the project."""

    DATASET = "dataset"
    TECHNICAL_DOCUMENT = "technical_document"
    HISTORICAL_CASES = "historical_cases"


class SourceManifest(BaseModel):
    """Provenance known before and after acquiring one external source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: SourceKind
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_url: HttpUrl
    download_url: HttpUrl
    version: str = Field(min_length=1)
    license_note: str = Field(min_length=1)
    expected_files: tuple[str, ...] = Field(min_length=1)
    retrieved_on: date | None = None
    archive_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_retrieval_evidence(self) -> Self:
        """Require retrieval date and checksum to be recorded together."""
        has_date = self.retrieved_on is not None
        has_checksum = self.archive_sha256 is not None
        if has_date != has_checksum:
            raise ValueError("retrieved_on and archive_sha256 must be provided together")
        return self


class TechnicalDocumentManifest(BaseModel):
    """Provenance for one independently acquired technical document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: Literal[SourceKind.TECHNICAL_DOCUMENT]
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    version: str = Field(min_length=1)
    chapter_number: int = Field(ge=1)
    chapter_title: str = Field(min_length=1)
    source_url: HttpUrl
    download_url: HttpUrl
    expected_filename: str = Field(pattern=r"^[^/\\]+\.pdf$")
    license_note: str = Field(min_length=1)
    retrieved_on: date | None = None
    file_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_complete_retrieval_evidence(self) -> Self:
        """Require retrieval date and file checksum to be recorded together."""
        has_date = self.retrieved_on is not None
        has_checksum = self.file_sha256 is not None
        if has_date != has_checksum:
            raise ValueError("retrieved_on and file_sha256 must be provided together")
        return self


def load_source_manifest(path: str | Path) -> SourceManifest:
    """Read a JSON file and validate it as a source manifest."""
    manifest_path = Path(path)
    raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return SourceManifest.model_validate(raw_data)


def load_technical_document_manifest(
    path: str | Path,
) -> TechnicalDocumentManifest:
    """Read JSON and validate it as one technical-document manifest."""
    manifest_path = Path(path)
    raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return TechnicalDocumentManifest.model_validate(raw_data)
