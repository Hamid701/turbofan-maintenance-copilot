"""Add the FD001 sensor_readings table.

Revision ID: 20260906_0004
Revises: 20260906_0003
Create Date: 2026-09-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260906_0004"
down_revision: str | Sequence[str] | None = "20260906_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FLOAT_COLUMNS = (
    *(f"setting_{number}" for number in range(1, 4)),
    *(f"sensor_{number}" for number in range(1, 22)),
)


def upgrade() -> None:
    """Create the engine-cycle observation table for FD001 train and test data."""
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("split", sa.String(length=5), nullable=False),
        sa.Column("unit_id", sa.SmallInteger(), nullable=False),
        sa.Column("cycle", sa.SmallInteger(), nullable=False),
        *(sa.Column(name, sa.Double(), nullable=False) for name in _FLOAT_COLUMNS),
        sa.UniqueConstraint(
            "split",
            "unit_id",
            "cycle",
            name="uq_sensor_readings_identity",
        ),
        sa.CheckConstraint(
            "split IN ('train', 'test')",
            name="ck_sensor_readings_split",
        ),
    )


def downgrade() -> None:
    """Drop the FD001 sensor_readings table."""
    op.drop_table("sensor_readings")
