"""SQLAlchemy models: the persisted retrieval corpus and the FD001 sensor readings."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Double,
    ForeignKey,
    Identity,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
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
