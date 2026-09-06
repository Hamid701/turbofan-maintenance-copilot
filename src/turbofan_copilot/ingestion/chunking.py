"""Page-contained text chunks for retrieval baselines."""

from pydantic import BaseModel, ConfigDict, Field

from turbofan_copilot.ingestion.document_page import ExtractedPage

DEFAULT_MAX_CHARACTERS = 1_500
DEFAULT_OVERLAP_CHARACTERS = 200


class DocumentChunk(BaseModel):
    """One retrieval unit with an exact single-page citation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    chapter_number: int = Field(ge=1)
    pdf_page_number: int = Field(ge=1)
    printed_page_label: str | None = Field(default=None, min_length=1)
    chunk_index: int = Field(ge=1)
    text: str = Field(min_length=1)


def chunk_page_text(
    page: ExtractedPage,
    normalized_text: str,
    *,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> tuple[DocumentChunk, ...]:
    """Split normalized page text into line-preserving, page-contained chunks."""
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")
    if overlap_characters < 0:
        raise ValueError("overlap_characters must be nonnegative")
    if overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be smaller than max_characters")
    if not normalized_text.strip():
        return ()

    lines = normalized_text.splitlines(keepends=True)
    base_ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(lines):
        budget = max_characters if not base_ranges else max_characters - overlap_characters
        end = start
        length = 0
        while end < len(lines):
            next_length = len(lines[end])
            if end > start and length + next_length > budget:
                break
            length += next_length
            end += 1
        base_ranges.append((start, end))
        start = end

    chunks = []
    for chunk_index, (start, end) in enumerate(base_ranges, start=1):
        overlap_start = start
        unique_length = sum(len(line) for line in lines[start:end])
        overlap_budget = min(
            overlap_characters,
            max(0, max_characters - unique_length),
        )
        while overlap_start > 0:
            previous_length = len(lines[overlap_start - 1])
            if previous_length > overlap_budget:
                break
            overlap_start -= 1
            overlap_budget -= previous_length

        chunks.append(
            DocumentChunk(
                source_id=page.source_id,
                chapter_number=page.chapter_number,
                pdf_page_number=page.pdf_page_number,
                printed_page_label=page.printed_page_label,
                chunk_index=chunk_index,
                text="".join(lines[overlap_start:end]),
            )
        )

    return tuple(chunks)
