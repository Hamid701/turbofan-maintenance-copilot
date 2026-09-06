"""Tests for the deterministic engine health-summary calculation."""

import pandas as pd
import pytest

from turbofan_copilot.health.engine_health import (
    EngineHealthSummary,
    summarise_engine_health,
)
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS, SETTING_COLUMNS


def engine_frame(
    sensor_series: dict[str, list[float]],
    *,
    cycles: int,
    split: str = "test",
    unit_id: int = 5,
) -> pd.DataFrame:
    """Build one engine's frame; unnamed sensors are held flat at zero."""
    rows = []
    for index in range(cycles):
        row: dict[str, object] = {"split": split, "unit_id": unit_id, "cycle": index + 1}
        for setting in SETTING_COLUMNS:
            row[setting] = 0.0
        for sensor in SENSOR_COLUMNS:
            row[sensor] = sensor_series.get(sensor, [0.0] * cycles)[index]
        rows.append(row)
    return pd.DataFrame(rows)


def test_linear_ramp_gives_the_expected_slope_and_change() -> None:
    frame = engine_frame({"sensor_5": [100.0, 102.0, 104.0, 106.0, 108.0]}, cycles=5)

    summary = summarise_engine_health(frame, window=30)

    trend = next(t for t in summary.trends if t.sensor == "sensor_5")
    assert trend.slope_per_cycle == pytest.approx(2.0)
    assert trend.total_change == pytest.approx(8.0)
    assert trend.window_mean == pytest.approx(104.0)
    assert trend.relative_change == pytest.approx(8.0 / 104.0)


def test_flat_sensor_has_a_zero_trend() -> None:
    frame = engine_frame({"sensor_3": [50.0] * 10}, cycles=10)

    trend = next(t for t in summarise_engine_health(frame).trends if t.sensor == "sensor_3")

    assert trend.slope_per_cycle == pytest.approx(0.0)
    assert trend.total_change == pytest.approx(0.0)
    assert trend.relative_change == pytest.approx(0.0)


def test_biggest_movers_are_ranked_by_relative_change() -> None:
    frame = engine_frame(
        {
            "sensor_11": [100.0 + index * 2.0 for index in range(10)],
            "sensor_2": [100.0 + index * 0.1 for index in range(10)],
        },
        cycles=10,
    )

    summary = summarise_engine_health(frame, top_movers=3)

    assert summary.biggest_movers[:2] == ("sensor_11", "sensor_2")
    assert len(summary.biggest_movers) == 3


def test_window_uses_only_the_most_recent_cycles() -> None:
    early = [10.0] * 40
    late = [10.0 + index for index in range(10)]
    frame = engine_frame({"sensor_7": early + late}, cycles=50)

    summary = summarise_engine_health(frame, window=10)

    assert summary.cycles_observed == 50
    assert summary.cycles_analysed == 10
    assert summary.first_analysed_cycle == 41
    assert summary.latest_cycle == 50
    trend = next(t for t in summary.trends if t.sensor == "sensor_7")
    assert trend.slope_per_cycle == pytest.approx(1.0)


def test_short_history_analyses_every_available_cycle() -> None:
    frame = engine_frame({"sensor_1": [1.0, 2.0, 3.0]}, cycles=3)

    summary = summarise_engine_health(frame, window=30)

    assert summary.cycles_observed == 3
    assert summary.cycles_analysed == 3


def test_one_trend_per_sensor_in_order_and_summary_is_frozen() -> None:
    summary = summarise_engine_health(engine_frame({}, cycles=5))

    assert tuple(trend.sensor for trend in summary.trends) == SENSOR_COLUMNS
    assert len(summary.trends) == 21
    with pytest.raises(ValueError, match="frozen"):
        summary.unit_id = 99


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="window must be at least 2"):
        summarise_engine_health(engine_frame({}, cycles=5), window=1)
    with pytest.raises(ValueError, match="top_movers must be at least 1"):
        summarise_engine_health(engine_frame({}, cycles=5), top_movers=0)
    with pytest.raises(ValueError, match="at least two cycles"):
        summarise_engine_health(engine_frame({}, cycles=1))

    mixed = pd.concat(
        [engine_frame({}, cycles=3, unit_id=1), engine_frame({}, cycles=3, unit_id=2)]
    )
    with pytest.raises(ValueError, match="exactly one"):
        summarise_engine_health(mixed)


def test_summary_model_forbids_extra_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        EngineHealthSummary.model_validate(
            {
                "split": "test",
                "unit_id": 1,
                "cycles_observed": 5,
                "cycles_analysed": 5,
                "first_analysed_cycle": 1,
                "latest_cycle": 5,
                "trends": (),
                "biggest_movers": (),
                "surprise": 1,
            }
        )
