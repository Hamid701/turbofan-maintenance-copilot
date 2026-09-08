"""Add the FD001 test-engine RUL target table.

Revision ID: 20260906_0005
Revises: 20260906_0004
Create Date: 2026-09-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260906_0005"
down_revision: str | Sequence[str] | None = "20260906_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ground-truth RUL table for FD001 test engines."""
    op.create_table(
        "engine_rul",
        sa.Column("unit_id", sa.SmallInteger(), primary_key=True),
        sa.Column("rul", sa.SmallInteger(), nullable=False),
        sa.CheckConstraint("rul >= 0", name="ck_engine_rul_nonnegative"),
    )


def downgrade() -> None:
    """Drop the RUL target table."""
    op.drop_table("engine_rul")
