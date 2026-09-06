"""Idempotent upsert of the retrieval corpus and its vectors into PostgreSQL."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from turbofan_copilot.db.models import Chunk, ChunkEmbedding
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.bge_embedder import EmbeddingVector

CHUNK_KEY = ("source_id", "pdf_page_number", "chunk_index")


@dataclass(frozen=True)
class IngestionResult:
    """Row counts in each table after an ingestion run."""

    chunk_rows: int
    embedding_rows: int


def ingest_chunks(
    session: Session,
    chunks: tuple[DocumentChunk, ...],
    vectors: tuple[EmbeddingVector, ...],
    *,
    model_id: str,
    model_revision: str,
) -> IngestionResult:
    """Upsert every chunk by its citation identity and every vector by chunk id.

    Running this twice with the same inputs changes nothing; running it with
    edited text or a new model revision updates the affected rows in place.
    """
    if not chunks:
        raise ValueError("ingest_chunks needs at least one chunk")
    if len(vectors) != len(chunks):
        raise ValueError("ingest_chunks needs exactly one vector per chunk")

    chunk_values = [
        {
            "source_id": chunk.source_id,
            "chapter_number": chunk.chapter_number,
            "pdf_page_number": chunk.pdf_page_number,
            "printed_page_label": chunk.printed_page_label,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    chunk_insert = insert(Chunk).values(chunk_values)
    session.execute(
        chunk_insert.on_conflict_do_update(
            index_elements=list(CHUNK_KEY),
            set_={
                "chapter_number": chunk_insert.excluded.chapter_number,
                "printed_page_label": chunk_insert.excluded.printed_page_label,
                "text": chunk_insert.excluded.text,
            },
        )
    )

    id_by_key = {
        (source_id, pdf_page_number, chunk_index): chunk_id
        for chunk_id, source_id, pdf_page_number, chunk_index in session.execute(
            select(
                Chunk.id,
                Chunk.source_id,
                Chunk.pdf_page_number,
                Chunk.chunk_index,
            )
        )
    }

    embedding_values = [
        {
            "chunk_id": id_by_key[(chunk.source_id, chunk.pdf_page_number, chunk.chunk_index)],
            "model_id": model_id,
            "model_revision": model_revision,
            "embedding": list(vector),
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    embedding_insert = insert(ChunkEmbedding).values(embedding_values)
    session.execute(
        embedding_insert.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                "model_id": embedding_insert.excluded.model_id,
                "model_revision": embedding_insert.excluded.model_revision,
                "embedding": embedding_insert.excluded.embedding,
            },
        )
    )
    session.flush()

    chunk_rows = session.scalar(select(func.count()).select_from(Chunk)) or 0
    embedding_rows = session.scalar(select(func.count()).select_from(ChunkEmbedding)) or 0
    return IngestionResult(chunk_rows=chunk_rows, embedding_rows=embedding_rows)
