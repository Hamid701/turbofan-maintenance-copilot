"""Tests for the RUL tool contract: which model answers, and what it declares."""

import pandas as pd
import pytest

from turbofan_copilot.health.degradation import (
    BASELINE_TEST_MAE,
    DegradationModel,
    SensorNorm,
)
from turbofan_copilot.health.rul_features import label_train_readings
from turbofan_copilot.health.rul_regressor import (
    BOOSTED_TEST_MAE,
    RulRegressor,
    fit_boosted_rul_regressor,
)
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS
from turbofan_copilot.llm.tools import RulPrediction, predict_rul

type Fitted = tuple[RulRegressor, DegradationModel]


def engine_frame(*, unit_id: int, cycles: int) -> pd.DataFrame:
    """One engine whose sensor_2 ramps with cycle; every other sensor is constant."""
    rows: list[dict[str, object]] = []
    for cycle in range(1, cycles + 1):
        row: dict[str, object] = {"split": "train", "unit_id": unit_id, "cycle": cycle}
        for sensor in SENSOR_COLUMNS:
            row[sensor] = 5.0 + (float(cycle) if sensor == "sensor_2" else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted() -> Fitted:
    """A boosted regressor plus a degradation model over the same synthetic engines."""
    engines = pd.concat([engine_frame(unit_id=unit, cycles=60 + unit) for unit in range(1, 9)])
    regressor = fit_boosted_rul_regressor(label_train_readings(engines), max_iter=40, window=30)
    baseline = DegradationModel(
        sensors=(SensorNorm(sensor="sensor_2", mean=0.0, std=1.0, sign=1.0),),
        failure_index=100.0,
    )
    return regressor, baseline


def test_a_full_history_is_answered_by_the_learned_model(fitted: Fitted) -> None:
    regressor, baseline = fitted

    prediction = predict_rul(
        engine_frame(unit_id=1, cycles=61), regressor=regressor, degradation_model=baseline
    )

    assert prediction.model == "boosted-trees"
    assert prediction.typical_error_cycles == BOOSTED_TEST_MAE
    assert prediction.cycles_observed == 61


def test_a_short_history_falls_back_instead_of_failing(fitted: Fitted) -> None:
    regressor, baseline = fitted
    # Below the 30-cycle feature window the learned model refuses; the caller must
    # still get an answer rather than an exception.
    short = engine_frame(unit_id=1, cycles=12)

    prediction = predict_rul(short, regressor=regressor, degradation_model=baseline)

    assert prediction.model == "degradation-index"
    assert prediction.typical_error_cycles == BASELINE_TEST_MAE
    assert prediction.cycles_observed == 12


def test_the_boundary_cycle_is_handled_by_the_learned_model(fitted: Fitted) -> None:
    regressor, baseline = fitted

    exactly_a_window = engine_frame(unit_id=1, cycles=regressor.window)
    prediction = predict_rul(exactly_a_window, regressor=regressor, degradation_model=baseline)

    assert prediction.model == "boosted-trees"


def test_the_prediction_declares_its_uncertainty_and_forbids_extras() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        RulPrediction.model_validate(
            {
                "estimated_rul": 40,
                "model": "boosted-trees",
                "typical_error_cycles": 8.4,
                "cycles_observed": 100,
                "surprise": 1,
            }
        )
    with pytest.raises(ValueError):
        RulPrediction.model_validate(
            {
                "estimated_rul": 40,
                "model": "a-model-we-never-measured",
                "typical_error_cycles": 8.4,
                "cycles_observed": 100,
            }
        )
