"""Add an HNSW cosine index on the chunk embeddings.

Revision ID: 20260906_0003
Revises: 20260906_0002
Create Date: 2026-09-06
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260906_0003"
down_revision: str | Sequence[str] | None = "20260906_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_chunk_embeddings_embedding_hnsw"


def upgrade() -> None:
    """Build the HNSW index used by cosine-distance nearest-neighbour queries."""
    op.create_index(
        INDEX_NAME,
        "chunk_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Drop the HNSW index; exact sequential-scan search still works without it."""
    op.drop_index(INDEX_NAME, table_name="chunk_embeddings")
