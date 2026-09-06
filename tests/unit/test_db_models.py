"""Structural checks for the persisted retrieval schema."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, UniqueConstraint

from turbofan_copilot.db.base import Base
from turbofan_copilot.db.models import Chunk, ChunkEmbedding, SensorReading
from turbofan_copilot.ingestion.fd001 import TRAJECTORY_COLUMNS
from turbofan_copilot.retrieval.bge_embedder import EMBEDDING_DIMENSION

CHUNKS = Base.metadata.tables["chunks"]
CHUNK_EMBEDDINGS = Base.metadata.tables["chunk_embeddings"]
SENSOR_READINGS = Base.metadata.tables["sensor_readings"]


def test_chunk_table_carries_the_document_chunk_fields() -> None:
    assert {column.name for column in CHUNKS.columns} == {
        "id",
        "source_id",
        "chapter_number",
        "pdf_page_number",
        "printed_page_label",
        "chunk_index",
        "text",
    }
    assert CHUNKS.columns["printed_page_label"].nullable is True
    assert CHUNKS.columns["source_id"].nullable is False
    assert CHUNKS.columns["text"].nullable is False


def test_chunk_table_has_the_natural_citation_key() -> None:
    unique_columns = {
        tuple(sorted(column.name for column in constraint.columns))
        for constraint in CHUNKS.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("chunk_index", "pdf_page_number", "source_id") in unique_columns


def test_embedding_column_matches_the_model_vector_contract() -> None:
    embedding_column = CHUNK_EMBEDDINGS.columns["embedding"]
    assert isinstance(embedding_column.type, Vector)
    assert embedding_column.type.dim == EMBEDDING_DIMENSION
    assert embedding_column.nullable is False


def test_embedding_is_registered_and_cascades_from_its_chunk() -> None:
    assert set(Base.metadata.tables) >= {"chunks", "chunk_embeddings"}
    foreign_key = next(iter(CHUNK_EMBEDDINGS.columns["chunk_id"].foreign_keys))
    assert foreign_key.column is CHUNKS.columns["id"]
    assert foreign_key.ondelete == "CASCADE"


def test_orm_classes_map_to_the_expected_tables() -> None:
    assert Chunk.__tablename__ == "chunks"
    assert ChunkEmbedding.__tablename__ == "chunk_embeddings"
    assert SensorReading.__tablename__ == "sensor_readings"


def test_sensor_readings_carries_the_fd001_trajectory_columns() -> None:
    value_columns = {column.name for column in SENSOR_READINGS.columns} - {"id", "split"}
    assert value_columns == set(TRAJECTORY_COLUMNS)
    assert all(not SENSOR_READINGS.columns[name].nullable for name in TRAJECTORY_COLUMNS)


def test_sensor_readings_has_the_natural_key_and_split_check() -> None:
    unique_columns = {
        tuple(sorted(column.name for column in constraint.columns))
        for constraint in SENSOR_READINGS.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("cycle", "split", "unit_id") in unique_columns

    checks = [
        str(constraint.sqltext)
        for constraint in SENSOR_READINGS.constraints
        if isinstance(constraint, CheckConstraint)
    ]
    assert any("split" in text for text in checks)
