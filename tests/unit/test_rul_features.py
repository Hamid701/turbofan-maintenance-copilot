"""Tests for windowed RUL features and the ridge regressor built on them."""

import numpy as np
import pandas as pd
import pytest

from turbofan_copilot.evaluation.rul_evaluation import label_train_readings
from turbofan_copilot.health.degradation import RUL_CAP
from turbofan_copilot.health.rul_features import (
    CYCLES_RUN,
    feature_names,
    features_from_window,
    informative_sensors,
    training_table,
    window_features,
)
from turbofan_copilot.health.rul_regressor import fit_boosted_rul_regressor, fit_rul_regressor
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS


def engine_frame(*, unit_id: int, cycles: int, ramp_per_cycle: float) -> pd.DataFrame:
    """One engine where sensor_2 ramps with cycle and every other sensor is constant."""
    rows: list[dict[str, object]] = []
    for cycle in range(1, cycles + 1):
        row: dict[str, object] = {"split": "train", "unit_id": unit_id, "cycle": cycle}
        for sensor in SENSOR_COLUMNS:
            row[sensor] = 5.0 + (ramp_per_cycle * cycle if sensor == "sensor_2" else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_informative_sensors_keeps_only_the_ones_that_move() -> None:
    assert informative_sensors(engine_frame(unit_id=1, cycles=40, ramp_per_cycle=2.0)) == (
        "sensor_2",
    )
    with pytest.raises(ValueError, match="no informative sensors"):
        informative_sensors(engine_frame(unit_id=1, cycles=40, ramp_per_cycle=0.0))


def test_feature_names_pair_each_sensor_with_four_descriptions() -> None:
    names = feature_names(("sensor_2", "sensor_7"))

    assert names[0] == CYCLES_RUN
    assert names[1:5] == ("sensor_2_last", "sensor_2_mean", "sensor_2_slope", "sensor_2_delta")
    assert len(names) == 1 + 2 * 4


def test_features_describe_a_known_ramp_exactly() -> None:
    # Values 10, 20, 30, 40, 50 at consecutive cycles: slope 10, mean 30, change 40.
    window = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])

    cycles_run, last, mean, slope, delta = features_from_window(window, ("sensor_2",), 105)

    assert cycles_run == 105.0
    assert last == 50.0
    assert mean == 30.0
    assert slope == pytest.approx(10.0)
    assert delta == 40.0


def test_features_reject_a_window_that_is_too_short_or_mis_shaped() -> None:
    with pytest.raises(ValueError, match="at least two cycles"):
        features_from_window(np.array([[1.0]]), ("sensor_2",), 1)
    with pytest.raises(ValueError, match="columns must match"):
        features_from_window(np.array([[1.0], [2.0]]), ("sensor_2", "sensor_3"), 2)


def test_window_features_use_only_the_most_recent_cycles() -> None:
    engine = engine_frame(unit_id=1, cycles=100, ramp_per_cycle=1.0)

    values = window_features(engine, ("sensor_2",), window=10)

    # Cycles 91..100 carry sensor values 96..105, so the window mean is 100.5.
    assert values[0] == 100.0
    assert values[1] == pytest.approx(105.0)
    assert values[2] == pytest.approx(100.5)
    assert values[3] == pytest.approx(1.0)


def test_training_table_skips_cycles_without_a_full_window() -> None:
    labelled = label_train_readings(engine_frame(unit_id=1, cycles=50, ramp_per_cycle=1.0))

    features, targets = training_table(labelled, ("sensor_2",), window=30)

    assert len(features) == len(targets) == 50 - 30 + 1
    assert list(features.columns) == list(feature_names(("sensor_2",)))
    # The last example is the engine's failure cycle, where nothing remains.
    assert features[CYCLES_RUN].iloc[-1] == 50.0
    assert targets.iloc[-1] == 0


def test_window_features_refuse_a_history_shorter_than_the_window() -> None:
    short = engine_frame(unit_id=1, cycles=29, ramp_per_cycle=1.0)

    with pytest.raises(ValueError, match="at least 30 cycles"):
        window_features(short, ("sensor_2",), window=30)

    # Exactly a full window is enough.
    assert window_features(
        engine_frame(unit_id=1, cycles=30, ramp_per_cycle=1.0), ("sensor_2",), window=30
    )


def test_training_table_requires_labels() -> None:
    with pytest.raises(ValueError, match="needs a rul column"):
        training_table(engine_frame(unit_id=1, cycles=50, ramp_per_cycle=1.0), ("sensor_2",))


def test_regressor_learns_a_clean_ramp_and_respects_the_output_contract() -> None:
    engines = pd.concat(
        [engine_frame(unit_id=unit, cycles=60 + unit, ramp_per_cycle=1.0) for unit in range(1, 9)]
    )

    regressor = fit_rul_regressor(label_train_readings(engines), alpha=0.1, window=30)
    late = regressor.predict(engines.loc[engines["unit_id"] == 1])

    assert regressor.sensors == ("sensor_2",)
    # Engine 1's last row is its failure, so almost nothing should remain.
    assert 0.0 <= late <= 10.0
    assert late == round(late)


def test_boosted_regressor_shares_the_ridge_output_contract() -> None:
    engines = pd.concat(
        [engine_frame(unit_id=unit, cycles=60 + unit, ramp_per_cycle=1.0) for unit in range(1, 9)]
    )
    labelled = label_train_readings(engines)

    boosted = fit_boosted_rul_regressor(labelled, learning_rate=0.1, max_iter=40, window=30)
    late = boosted.predict(engines.loc[engines["unit_id"] == 1])

    assert boosted.sensors == ("sensor_2",)
    assert 0.0 <= late <= float(RUL_CAP)
    assert late == round(late)


def test_boosted_regressor_is_deterministic() -> None:
    engines = pd.concat(
        [engine_frame(unit_id=unit, cycles=60 + unit, ramp_per_cycle=1.0) for unit in range(1, 9)]
    )
    labelled = label_train_readings(engines)
    history = engines.loc[engines["unit_id"] == 3]

    first = fit_boosted_rul_regressor(labelled, max_iter=40, window=30).predict(history)
    second = fit_boosted_rul_regressor(labelled, max_iter=40, window=30).predict(history)

    assert first == second


def test_regressor_clips_predictions_into_the_target_range() -> None:
    engines = pd.concat(
        [engine_frame(unit_id=unit, cycles=60 + unit, ramp_per_cycle=1.0) for unit in range(1, 9)]
    )
    regressor = fit_rul_regressor(label_train_readings(engines), alpha=0.1, window=30)

    # A wildly out-of-range history must still produce a usable number.
    absurd = engine_frame(unit_id=99, cycles=40, ramp_per_cycle=-500.0)
    prediction = regressor.predict(absurd)

    assert 0.0 <= prediction <= float(RUL_CAP)
