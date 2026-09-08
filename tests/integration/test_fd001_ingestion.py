"""Live-database checks for the idempotent FD001 sensor-reading ingestion."""

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import ingest_fd001_readings, ingest_fd001_rul
from turbofan_copilot.db.models import EngineRul, SensorReading
from turbofan_copilot.ingestion.fd001 import TRAJECTORY_COLUMNS

pytestmark = pytest.mark.integration

PROBE_UNITS = (901, 902)


def probe_frame(sensor_1_value: float = 1.0) -> pd.DataFrame:
    """Build a small contiguous trajectory frame under probe unit ids."""
    rows = []
    for offset, unit in enumerate(PROBE_UNITS, start=1):
        for cycle in (1, 2, 3):
            settings = [0.1, 0.2, 100.0]
            sensors = [sensor_1_value, *([float(cycle + offset)] * 20)]
            rows.append([unit, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS))


def probe_rows(session: Session, split: str) -> list[SensorReading]:
    """Return the probe rows in the session, ordered by unit then cycle."""
    return list(
        session.execute(
            select(SensorReading)
            .where(SensorReading.split == split, SensorReading.unit_id.in_(PROBE_UNITS))
            .order_by(SensorReading.unit_id, SensorReading.cycle)
        ).scalars()
    )


def test_ingest_writes_rows_with_values_intact(db_session: Session) -> None:
    ingest_fd001_readings(db_session, probe_frame(sensor_1_value=7.5), split="train")

    rows = probe_rows(db_session, "train")
    assert len(rows) == 6
    assert [(row.unit_id, row.cycle) for row in rows] == [
        (901, 1),
        (901, 2),
        (901, 3),
        (902, 1),
        (902, 2),
        (902, 3),
    ]
    assert rows[0].sensor_1 == pytest.approx(7.5)
    assert rows[0].setting_3 == pytest.approx(100.0)


def test_reingest_updates_in_place_without_duplicates(db_session: Session) -> None:
    ingest_fd001_readings(db_session, probe_frame(sensor_1_value=1.0), split="train")
    ingest_fd001_readings(db_session, probe_frame(sensor_1_value=2.0), split="train")

    rows = probe_rows(db_session, "train")
    assert len(rows) == 6
    assert all(row.sensor_1 == pytest.approx(2.0) for row in rows)


def test_same_unit_id_is_independent_across_splits(db_session: Session) -> None:
    ingest_fd001_readings(db_session, probe_frame(sensor_1_value=1.0), split="train")
    ingest_fd001_readings(db_session, probe_frame(sensor_1_value=9.0), split="test")

    assert len(probe_rows(db_session, "train")) == 6
    assert len(probe_rows(db_session, "test")) == 6
    assert probe_rows(db_session, "test")[0].sensor_1 == pytest.approx(9.0)


def test_ingest_rejects_an_unknown_split(db_session: Session) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        ingest_fd001_readings(db_session, probe_frame(), split="validation")  # type: ignore[arg-type]


def stored_rul(session: Session, unit_id: int) -> int | None:
    """Return the persisted RUL target for one unit, or None."""
    return session.scalar(select(EngineRul.rul).where(EngineRul.unit_id == unit_id))


def test_rul_targets_upsert_by_unit_and_reject_negatives(db_session: Session) -> None:
    ingest_fd001_rul(db_session, pd.Series([112, 98, 5], name="rul"))
    assert [stored_rul(db_session, unit) for unit in (1, 2, 3)] == [112, 98, 5]

    ingest_fd001_rul(db_session, pd.Series([120, 98, 5], name="rul"))
    assert stored_rul(db_session, 1) == 120

    with pytest.raises(ValueError, match="must not be negative"):
        ingest_fd001_rul(db_session, pd.Series([10, -1], name="rul"))
