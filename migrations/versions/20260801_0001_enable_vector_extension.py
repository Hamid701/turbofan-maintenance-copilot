"""Enable the pgvector extension.

Revision ID: 20260801_0001
Revises:
Create Date: 2026-08-01
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260801_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Make the vector type and vector indexes available to later migrations."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Remove pgvector only when no dependent database objects remain."""
    op.execute("DROP EXTENSION IF EXISTS vector")
