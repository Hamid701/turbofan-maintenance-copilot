"""Add the query_runs and feedback tables.

Revision ID: 20260907_0006
Revises: 20260906_0005
Create Date: 2026-09-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260907_0006"
down_revision: str | Sequence[str] | None = "20260906_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the served-query log and the feedback table."""
    op.create_table(
        "query_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("citation_count", sa.SmallInteger(), nullable=False),
        sa.Column("citations", JSONB(), nullable=False),
        sa.Column("engine_unit_id", sa.SmallInteger(), nullable=True),
        sa.Column("answer_model", sa.String(length=64), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_query_runs_request_id", "query_runs", ["request_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("query_request_id", sa.String(length=128), nullable=False),
        sa.Column("rating", sa.String(length=4), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("rating IN ('up', 'down')", name="ck_feedback_rating"),
    )
    op.create_index("ix_feedback_query_request_id", "feedback", ["query_request_id"])


def downgrade() -> None:
    """Drop the feedback and query_runs tables."""
    op.drop_index("ix_feedback_query_request_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_query_runs_request_id", table_name="query_runs")
    op.drop_table("query_runs")
