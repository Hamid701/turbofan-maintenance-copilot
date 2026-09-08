"""Read and idempotently write FD001 rows in the ``sensor_readings`` table."""

from collections.abc import Sequence
from typing import Literal, cast

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from turbofan_copilot.db.models import (
    SENSOR_READING_SPLITS,
    EngineRul,
    SensorReading,
)
from turbofan_copilot.ingestion.fd001 import (
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    TRAJECTORY_COLUMNS,
)

Split = Literal["train", "test"]

_VALUE_COLUMNS = (*SETTING_COLUMNS, *SENSOR_COLUMNS)
_READING_COLUMNS = ("split", "unit_id", "cycle", *_VALUE_COLUMNS)
_BATCH_ROWS = 2000


def ingest_fd001_readings(session: Session, frame: pd.DataFrame, *, split: Split) -> int:
    """Upsert one trajectory frame and return the row count for that split.

    The natural key is ``(split, unit_id, cycle)``, so re-running with the same file
    is a no-op and re-running with corrected values updates the affected rows. This
    guards its own inputs only; the caller runs ``validate_trajectory_structure``.
    """
    if split not in SENSOR_READING_SPLITS:
        raise ValueError(f"split must be one of {SENSOR_READING_SPLITS}; got {split!r}")
    if tuple(frame.columns) != TRAJECTORY_COLUMNS:
        raise ValueError("frame columns do not match the FD001 trajectory contract")
    if frame.empty:
        raise ValueError("frame must not be empty")
    if bool(frame.isna().to_numpy().any()):
        raise ValueError("frame contains missing values")

    rows: list[dict[str, object]] = []
    for values in frame[list(TRAJECTORY_COLUMNS)].itertuples(index=False, name=None):
        record = dict(zip(TRAJECTORY_COLUMNS, values, strict=True))
        row: dict[str, object] = {
            "split": split,
            "unit_id": int(cast(int, record["unit_id"])),
            "cycle": int(cast(int, record["cycle"])),
        }
        for column in _VALUE_COLUMNS:
            row[column] = float(cast(float, record[column]))
        rows.append(row)

    for start in range(0, len(rows), _BATCH_ROWS):
        batch = rows[start : start + _BATCH_ROWS]
        statement = insert(SensorReading).values(batch)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["split", "unit_id", "cycle"],
                set_={column: statement.excluded[column] for column in _VALUE_COLUMNS},
            )
        )
    session.flush()

    count = session.scalar(
        select(func.count()).select_from(SensorReading).where(SensorReading.split == split)
    )
    return count or 0


def ingest_fd001_rul(session: Session, rul: pd.Series) -> int:
    """Upsert the FD001 test-engine RUL targets and return the row count.

    ``rul`` is the one-column series from ``load_rul_file``; row ``i`` is test
    engine ``i + 1``. The primary key is ``unit_id``, so re-running is a no-op.
    """
    if rul.empty:
        raise ValueError("rul series must not be empty")
    if bool((rul < 0).any()):
        raise ValueError("rul targets must not be negative")

    rows = [{"unit_id": index + 1, "rul": int(value)} for index, value in enumerate(rul)]
    statement = insert(EngineRul).values(rows)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["unit_id"],
            set_={"rul": statement.excluded["rul"]},
        )
    )
    session.flush()
    return session.scalar(select(func.count()).select_from(EngineRul)) or 0


def _readings_frame(rows: Sequence[SensorReading]) -> pd.DataFrame:
    """Turn SensorReading ORM rows into a DataFrame with the reading columns."""
    return pd.DataFrame(
        [{column: getattr(row, column) for column in _READING_COLUMNS} for row in rows]
    )


def load_engine_readings(session: Session, split: str, unit_id: int) -> pd.DataFrame:
    """Return one engine's readings ordered by cycle, as a DataFrame."""
    rows = (
        session.execute(
            select(SensorReading)
            .where(SensorReading.split == split, SensorReading.unit_id == unit_id)
            .order_by(SensorReading.cycle)
        )
        .scalars()
        .all()
    )
    if not rows:
        raise LookupError(f"no readings for engine {unit_id} in split {split!r}")
    return _readings_frame(rows)


def load_split_readings(session: Session, split: str) -> pd.DataFrame:
    """Return every reading for one split, ordered by unit then cycle, as a DataFrame."""
    rows = (
        session.execute(
            select(SensorReading)
            .where(SensorReading.split == split)
            .order_by(SensorReading.unit_id, SensorReading.cycle)
        )
        .scalars()
        .all()
    )
    if not rows:
        raise LookupError(f"no readings for split {split!r}")
    return _readings_frame(rows)
