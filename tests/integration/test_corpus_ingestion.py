"""Live-database checks for the idempotent corpus ingestion."""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.ingestion import ingest_chunks
from turbofan_copilot.db.models import Chunk, ChunkEmbedding
from turbofan_copilot.ingestion.chunking import DocumentChunk

pytestmark = pytest.mark.integration

PROBE_SOURCE = "integration-probe"


def probe_chunk(chunk_index: int, *, text: str) -> DocumentChunk:
    """Return one chunk under a source id that cannot collide with the FAA corpus."""
    return DocumentChunk(
        source_id=PROBE_SOURCE,
        chapter_number=9,
        pdf_page_number=99,
        printed_page_label=None,
        chunk_index=chunk_index,
        text=text,
    )


def probe_chunks(session: Session) -> list[Chunk]:
    """Return the probe rows currently visible in the session, ordered by chunk index."""
    return list(
        session.execute(
            select(Chunk).where(Chunk.source_id == PROBE_SOURCE).order_by(Chunk.chunk_index)
        ).scalars()
    )


def test_ingest_writes_chunks_and_vectors(db_session: Session) -> None:
    chunks = (probe_chunk(1, text="first"), probe_chunk(2, text="second"))
    vectors = (tuple([0.1] * 384), tuple([0.2] * 384))

    ingest_chunks(db_session, chunks, vectors, model_id="probe", model_revision="r1")

    rows = probe_chunks(db_session)
    assert [row.text for row in rows] == ["first", "second"]
    embedding = db_session.get(ChunkEmbedding, rows[0].id)
    assert embedding is not None
    assert embedding.model_revision == "r1"
    assert len(list(embedding.embedding)) == 384


def test_reingesting_updates_in_place_without_duplicates(db_session: Session) -> None:
    vector = tuple([0.1] * 384)
    ingest_chunks(
        db_session, (probe_chunk(1, text="v1"),), (vector,), model_id="probe", model_revision="r1"
    )
    ingest_chunks(
        db_session, (probe_chunk(1, text="v2"),), (vector,), model_id="probe", model_revision="r2"
    )

    rows = probe_chunks(db_session)
    assert len(rows) == 1
    assert rows[0].text == "v2"
    embedding = db_session.get(ChunkEmbedding, rows[0].id)
    assert embedding is not None
    assert embedding.model_revision == "r2"


def test_ingest_rejects_mismatched_inputs(db_session: Session) -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        ingest_chunks(db_session, (), (), model_id="probe", model_revision="r1")
    with pytest.raises(ValueError, match="one vector per chunk"):
        ingest_chunks(
            db_session, (probe_chunk(1, text="x"),), (), model_id="probe", model_revision="r1"
        )
