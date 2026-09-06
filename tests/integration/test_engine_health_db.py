"""Live-database checks for reading an engine and summarising its health."""

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import ingest_fd001_readings, load_engine_readings
from turbofan_copilot.db.models import SensorReading
from turbofan_copilot.health.engine_health import summarise_engine_health
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS, SETTING_COLUMNS, TRAJECTORY_COLUMNS

pytestmark = pytest.mark.integration

PROBE_UNIT = 903


def probe_frame(ramp_sensor: str) -> pd.DataFrame:
    """One probe engine with a rising ramp on ``ramp_sensor`` and flat elsewhere."""
    rows = []
    for cycle in range(1, 11):
        row = [PROBE_UNIT, cycle, *([0.0] * len(SETTING_COLUMNS))]
        row += [float(cycle * 3) if name == ramp_sensor else 5.0 for name in SENSOR_COLUMNS]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS))


def test_load_engine_readings_returns_ordered_rows(db_session: Session) -> None:
    ingest_fd001_readings(db_session, probe_frame("sensor_9"), split="train")

    frame = load_engine_readings(db_session, "train", PROBE_UNIT)

    assert list(frame["cycle"]) == list(range(1, 11))
    assert set(frame.columns) == {"split", "unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS}
    assert (frame["unit_id"] == PROBE_UNIT).all()


def test_load_engine_readings_raises_for_an_unknown_engine(db_session: Session) -> None:
    with pytest.raises(LookupError, match="no readings for engine"):
        load_engine_readings(db_session, "train", 999)


def test_summary_over_persisted_readings(db_session: Session) -> None:
    ingest_fd001_readings(db_session, probe_frame("sensor_9"), split="train")
    frame = load_engine_readings(db_session, "train", PROBE_UNIT)

    summary = summarise_engine_health(frame, window=10)

    assert summary.split == "train"
    assert summary.unit_id == PROBE_UNIT
    assert summary.cycles_observed == 10
    assert summary.biggest_movers[0] == "sensor_9"
    trend = next(t for t in summary.trends if t.sensor == "sensor_9")
    assert trend.slope_per_cycle == pytest.approx(3.0)


def test_real_engine_one_summary(db_session: Session) -> None:
    total = db_session.scalar(
        select(SensorReading.id).where(SensorReading.split == "train", SensorReading.unit_id == 1)
    )
    if total is None:
        pytest.skip("FD001 not ingested; run scripts/ingest_fd001.py")

    frame = load_engine_readings(db_session, "train", 1)
    summary = summarise_engine_health(frame)

    assert summary.unit_id == 1
    assert summary.cycles_observed == 192
    assert summary.latest_cycle == 192
    assert len(summary.trends) == 21
