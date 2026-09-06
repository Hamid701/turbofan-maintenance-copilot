"""Read the persisted retrieval corpus back out of PostgreSQL."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.models import Chunk, ChunkEmbedding
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.bge_embedder import EmbeddingVector


@dataclass(frozen=True)
class PersistedCorpus:
    """Every stored chunk with its aligned vector, plus the model that produced them."""

    chunks: tuple[DocumentChunk, ...]
    vectors: tuple[EmbeddingVector, ...]
    model_id: str
    model_revision: str


def load_persisted_corpus(session: Session) -> PersistedCorpus:
    """Load all chunks and vectors in deterministic citation order, aligned per row."""
    rows = session.execute(
        select(Chunk, ChunkEmbedding)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .order_by(Chunk.source_id, Chunk.pdf_page_number, Chunk.chunk_index)
    ).all()
    if not rows:
        raise RuntimeError("no persisted corpus found; run scripts/ingest_700_char_corpus.py first")

    chunks = tuple(
        DocumentChunk(
            source_id=chunk.source_id,
            chapter_number=chunk.chapter_number,
            pdf_page_number=chunk.pdf_page_number,
            printed_page_label=chunk.printed_page_label,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )
        for chunk, _ in rows
    )
    vectors = tuple(tuple(float(value) for value in embedding.embedding) for _, embedding in rows)

    models = {(embedding.model_id, embedding.model_revision) for _, embedding in rows}
    if len(models) != 1:
        raise RuntimeError(f"persisted corpus mixes embedding models: {sorted(models)}")
    model_id, model_revision = next(iter(models))

    return PersistedCorpus(
        chunks=chunks,
        vectors=vectors,
        model_id=model_id,
        model_revision=model_revision,
    )
