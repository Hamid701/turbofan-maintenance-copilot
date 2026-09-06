"""Normalized page records produced by technical-document extraction."""

from pydantic import BaseModel, ConfigDict, Field


class ExtractedPage(BaseModel):
    """Text and citation identity for one physical PDF page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    chapter_number: int = Field(ge=1)
    pdf_page_number: int = Field(ge=1)
    printed_page_label: str | None = Field(default=None, min_length=1)
    text: str
