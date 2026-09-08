"""SQLAlchemy models: the persisted corpus, FD001 readings, and served-query logs."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Identity,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from turbofan_copilot.db.base import Base
from turbofan_copilot.retrieval.bge_embedder import EMBEDDING_DIMENSION


class Chunk(Base):
    """One page-contained retrieval unit with its exact citation identity."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "pdf_page_number",
            "chunk_index",
            name="uq_chunks_identity",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chapter_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    pdf_page_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    printed_page_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunk_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped["ChunkEmbedding | None"] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        uselist=False,
    )


EMBEDDING_HNSW_INDEX = "ix_chunk_embeddings_embedding_hnsw"


class ChunkEmbedding(Base):
    """The BGE vector for one chunk, kept apart so it can be rebuilt per model."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        Index(
            EMBEDDING_HNSW_INDEX,
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")


SENSOR_READING_SPLITS = ("train", "test")


class SensorReading(Base):
    """One FD001 engine-cycle observation: three operating settings and 21 sensors.

    The C-MAPSS sensors are anonymous, so they stay numbered; no physical meaning
    is attached here. ``split`` separates the run-to-failure ``train`` engines from
    the truncated ``test`` engines, whose unit ids overlap.
    """

    __tablename__ = "sensor_readings"
    __table_args__ = (
        UniqueConstraint("split", "unit_id", "cycle", name="uq_sensor_readings_identity"),
        CheckConstraint(
            "split IN ('train', 'test')",
            name="ck_sensor_readings_split",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    split: Mapped[str] = mapped_column(String(5), nullable=False)
    unit_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    cycle: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    setting_1: Mapped[float] = mapped_column(Double, nullable=False)
    setting_2: Mapped[float] = mapped_column(Double, nullable=False)
    setting_3: Mapped[float] = mapped_column(Double, nullable=False)

    sensor_1: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_2: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_3: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_4: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_5: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_6: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_7: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_8: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_9: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_10: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_11: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_12: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_13: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_14: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_15: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_16: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_17: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_18: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_19: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_20: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_21: Mapped[float] = mapped_column(Double, nullable=False)


class EngineRul(Base):
    """The ground-truth remaining-useful-life target for one FD001 test engine.

    From ``RUL_FD001.txt``: the cycles each truncated test engine still has left
    after its last observed cycle. Train engines have no target (they run to failure).
    """

    __tablename__ = "engine_rul"
    __table_args__ = (CheckConstraint("rul >= 0", name="ck_engine_rul_nonnegative"),)

    unit_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    rul: Mapped[int] = mapped_column(SmallInteger, nullable=False)


FEEDBACK_RATINGS = ("up", "down")


class QueryRun(Base):
    """One answered ``POST /v1/query`` request, for tracing and later analysis."""

    __tablename__ = "query_runs"
    __table_args__ = (Index("ix_query_runs_request_id", "request_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    abstained: Mapped[bool] = mapped_column(nullable=False)
    citation_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    citations: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    engine_unit_id: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    answer_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Feedback(Base):
    """A reader's up/down verdict on an earlier answer, keyed by its request id."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN ('up', 'down')", name="ck_feedback_rating"),
        Index("ix_feedback_query_request_id", "query_request_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    query_request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    rating: Mapped[str] = mapped_column(String(4), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
