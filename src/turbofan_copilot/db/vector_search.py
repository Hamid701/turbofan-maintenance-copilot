"""Cosine nearest-neighbour search over the persisted embeddings, in SQL."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.models import Chunk, ChunkEmbedding
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder, EmbeddingVector

type ScoredChunk = tuple[float, DocumentChunk]


def _row_to_chunk(chunk: Chunk) -> DocumentChunk:
    """Rebuild the ingestion-time chunk contract from one row."""
    return DocumentChunk(
        source_id=chunk.source_id,
        chapter_number=chunk.chapter_number,
        pdf_page_number=chunk.pdf_page_number,
        printed_page_label=chunk.printed_page_label,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
    )


def search_by_vector(
    session: Session,
    query_vector: EmbeddingVector,
    *,
    limit: int,
    exact: bool = False,
) -> tuple[ScoredChunk, ...]:
    """Return the ``limit`` closest chunks by cosine similarity.

    ``exact=True`` sorts every row (a sequential scan) and is the reference result.
    ``exact=False`` lets PostgreSQL use the HNSW index, which is approximate.
    """
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    distance = ChunkEmbedding.embedding.cosine_distance(list(query_vector))
    statement = (
        select(Chunk, distance.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .order_by(
            "distance",
            Chunk.source_id,
            Chunk.pdf_page_number,
            Chunk.chunk_index,
        )
    )
    if exact:
        rows = list(session.execute(statement).all())[:limit]
    else:
        rows = list(session.execute(statement.limit(limit)).all())

    return tuple((1.0 - float(row_distance), _row_to_chunk(chunk)) for chunk, row_distance in rows)


class DbSemanticRetriever:
    """Rank the persisted corpus by cosine similarity, computed in PostgreSQL."""

    def __init__(self, session: Session, embedder: BgeEmbedder, *, exact: bool = True) -> None:
        self._session = session
        self._embedder = embedder
        self._exact = exact

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        """Return the highest cosine-similarity chunks for the embedded question."""
        query_vector = self._embedder.embed_question(question)
        return search_by_vector(
            self._session,
            query_vector,
            limit=limit,
            exact=self._exact,
        )
