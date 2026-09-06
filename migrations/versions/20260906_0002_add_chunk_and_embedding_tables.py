"""Add the persisted retrieval corpus tables.

Revision ID: 20260906_0002
Revises: 20260801_0001
Create Date: 2026-09-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from turbofan_copilot.retrieval.bge_embedder import EMBEDDING_DIMENSION

revision: str = "20260906_0002"
down_revision: str | Sequence[str] | None = "20260801_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the chunk and chunk-embedding tables."""
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("chapter_number", sa.SmallInteger(), nullable=False),
        sa.Column("pdf_page_number", sa.SmallInteger(), nullable=False),
        sa.Column("printed_page_label", sa.String(length=32), nullable=True),
        sa.Column("chunk_index", sa.SmallInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "source_id",
            "pdf_page_number",
            "chunk_index",
            name="uq_chunks_identity",
        ),
    )
    op.create_table(
        "chunk_embeddings",
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_revision", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
    )


def downgrade() -> None:
    """Drop the chunk and chunk-embedding tables."""
    op.drop_table("chunk_embeddings")
    op.drop_table("chunks")
