"""Tests for the deterministic degradation-index RUL baseline."""

import pandas as pd
import pytest

from turbofan_copilot.health.degradation import (
    RUL_CAP,
    DegradationModel,
    RulEstimate,
    SensorNorm,
    estimate_engine_rul,
    fit_degradation_model,
)
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS


def train_frame(
    *,
    engines: int,
    cycles: int,
    ramp_sensor: str,
    ramp_per_cycle: float,
) -> pd.DataFrame:
    """Train data where one sensor ramps with cycle and the rest are constant."""
    rows = []
    for unit in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            row: dict[str, object] = {"split": "train", "unit_id": unit, "cycle": cycle}
            for sensor in SENSOR_COLUMNS:
                row[sensor] = 5.0 + (ramp_per_cycle * cycle if sensor == ramp_sensor else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def one_sensor_engine(
    values: list[float], *, split: str = "test", unit_id: int = 7
) -> pd.DataFrame:
    """One engine's frame carrying only sensor_1, cycles 1..n."""
    return pd.DataFrame(
        {
            "split": split,
            "unit_id": unit_id,
            "cycle": list(range(1, len(values) + 1)),
            "sensor_1": values,
        }
    )


def test_fit_keeps_only_varying_sensors_and_aligns_their_sign() -> None:
    rising = fit_degradation_model(
        train_frame(engines=2, cycles=10, ramp_sensor="sensor_4", ramp_per_cycle=2.0)
    )
    assert [norm.sensor for norm in rising.sensors] == ["sensor_4"]
    assert rising.sensors[0].sign == 1.0

    falling = fit_degradation_model(
        train_frame(engines=2, cycles=10, ramp_sensor="sensor_9", ramp_per_cycle=-3.0)
    )
    assert falling.sensors[0].sign == -1.0


def test_fit_raises_when_no_sensor_varies() -> None:
    flat = train_frame(engines=2, cycles=5, ramp_sensor="sensor_1", ramp_per_cycle=0.0)
    with pytest.raises(ValueError, match="no informative sensors"):
        fit_degradation_model(flat)


def test_estimate_extrapolates_linear_wear_to_the_failure_threshold() -> None:
    model = DegradationModel(
        sensors=(SensorNorm(sensor="sensor_1", mean=0.0, std=1.0, sign=1.0),),
        failure_index=100.0,
    )

    estimate = estimate_engine_rul(one_sensor_engine([10.0, 20.0, 30.0, 40.0, 50.0]), model)

    assert estimate.index_slope_per_cycle == pytest.approx(10.0)
    assert estimate.current_index == pytest.approx(50.0)
    assert estimate.raw_estimate == pytest.approx(5.0)
    assert estimate.estimated_rul == 5
    assert estimate.split == "test"
    assert estimate.unit_id == 7


def test_estimate_returns_the_cap_when_no_degradation_is_detected() -> None:
    model = DegradationModel(
        sensors=(SensorNorm(sensor="sensor_1", mean=0.0, std=1.0, sign=1.0),),
        failure_index=100.0,
    )

    estimate = estimate_engine_rul(one_sensor_engine([5.0, 5.0, 5.0, 5.0]), model)

    assert estimate.index_slope_per_cycle == pytest.approx(0.0)
    assert estimate.estimated_rul == RUL_CAP


def test_estimate_returns_zero_when_already_past_the_threshold() -> None:
    model = DegradationModel(
        sensors=(SensorNorm(sensor="sensor_1", mean=0.0, std=1.0, sign=1.0),),
        failure_index=10.0,
    )

    estimate = estimate_engine_rul(one_sensor_engine([50.0, 60.0, 70.0]), model)

    assert estimate.raw_estimate < 0
    assert estimate.estimated_rul == 0


def test_estimate_clips_a_huge_extrapolation_to_the_cap() -> None:
    model = DegradationModel(
        sensors=(SensorNorm(sensor="sensor_1", mean=0.0, std=1.0, sign=1.0),),
        failure_index=1_000_000.0,
    )

    estimate = estimate_engine_rul(one_sensor_engine([1.0, 1.01, 1.02, 1.03]), model)

    assert estimate.estimated_rul == RUL_CAP


def test_estimate_rejects_short_history_and_bad_window() -> None:
    model = DegradationModel(
        sensors=(SensorNorm(sensor="sensor_1", mean=0.0, std=1.0, sign=1.0),),
        failure_index=100.0,
    )
    with pytest.raises(ValueError, match="window must be at least 2"):
        estimate_engine_rul(one_sensor_engine([1.0, 2.0, 3.0]), model, window=1)
    with pytest.raises(ValueError, match="at least two cycles"):
        estimate_engine_rul(one_sensor_engine([1.0]), model)


def test_rul_estimate_model_forbids_extra_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        RulEstimate.model_validate(
            {
                "split": "test",
                "unit_id": 1,
                "latest_cycle": 5,
                "cycles_analysed": 5,
                "current_index": 0.0,
                "index_slope_per_cycle": 0.0,
                "raw_estimate": 0.0,
                "estimated_rul": 0,
                "surprise": 1,
            }
        )
