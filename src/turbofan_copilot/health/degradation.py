"""Deterministic degradation-index RUL baseline for FD001 engines.

Pure calculation: no database, no LLM. The model is fitted on the train set and
turns one engine's recent cycles into an estimated remaining-useful-life.
"""

from statistics import correlation, fmean, linear_regression, pstdev

import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbofan_copilot.health.engine_health import DEFAULT_WINDOW
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS

STD_FLOOR = 1e-6
RUL_CAP = 125


class SensorNorm(BaseModel):
    """How to standardise and sign-align one informative sensor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sensor: str
    mean: float
    std: float
    sign: float


class DegradationModel(BaseModel):
    """Fitted on the train set: how to turn one cycle's sensors into a wear index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sensors: tuple[SensorNorm, ...]
    failure_index: float


class RulEstimate(BaseModel):
    """One engine's estimated remaining cycles, and the numbers behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    unit_id: int
    latest_cycle: int
    cycles_analysed: int
    current_index: float
    index_slope_per_cycle: float
    raw_estimate: float
    estimated_rul: int


def _index_per_cycle(frame: pd.DataFrame, sensors: tuple[SensorNorm, ...]) -> list[float]:
    """Sign-aligned mean of the z-scored informative sensors, one value per row."""
    columns = [norm.sensor for norm in sensors]
    values: list[float] = []
    for row in frame[columns].itertuples(index=False, name=None):
        values.append(
            fmean(
                norm.sign * (float(value) - norm.mean) / norm.std
                for norm, value in zip(sensors, row, strict=True)
            )
        )
    return values


def _require_single_engine(readings: pd.DataFrame) -> tuple[str, int]:
    """Return the one (split, unit_id) in the frame, or raise."""
    splits = set(readings["split"])
    units = set(readings["unit_id"])
    if len(splits) != 1 or len(units) != 1:
        raise ValueError("readings must contain exactly one (split, unit_id)")
    return str(next(iter(splits))), int(next(iter(units)))


def fit_degradation_model(train_readings: pd.DataFrame) -> DegradationModel:
    """Fit the wear index on every run-to-failure train engine."""
    ordered = train_readings.sort_values(["unit_id", "cycle"])
    cycles = [int(value) for value in ordered["cycle"].tolist()]

    sensors: list[SensorNorm] = []
    for name in SENSOR_COLUMNS:
        column = [float(value) for value in ordered[name].tolist()]
        std = pstdev(column)
        if std <= STD_FLOOR:
            continue
        sign = 1.0 if correlation(column, cycles) >= 0 else -1.0
        sensors.append(SensorNorm(sensor=name, mean=fmean(column), std=std, sign=sign))
    if not sensors:
        raise ValueError("no informative sensors found in the training data")

    fitted = tuple(sensors)
    index = _index_per_cycle(ordered, fitted)
    last_index: dict[int, float] = {}
    for unit, value in zip((int(unit) for unit in ordered["unit_id"].tolist()), index, strict=True):
        last_index[unit] = value
    return DegradationModel(sensors=fitted, failure_index=fmean(last_index.values()))


def estimate_engine_rul(
    engine_readings: pd.DataFrame,
    model: DegradationModel,
    *,
    window: int = DEFAULT_WINDOW,
) -> RulEstimate:
    """Estimate one engine's remaining cycles by extrapolating its wear index."""
    if window < 2:
        raise ValueError("window must be at least 2 cycles")
    split, unit_id = _require_single_engine(engine_readings)

    ordered = engine_readings.sort_values("cycle")
    if len(ordered) < 2:
        raise ValueError("an engine needs at least two cycles for an estimate")

    analysed = ordered.tail(window)
    cycles = [int(value) for value in analysed["cycle"].tolist()]
    index = _index_per_cycle(analysed, model.sensors)
    fit = linear_regression(cycles, index)
    current_index = fit.slope * cycles[-1] + fit.intercept

    if fit.slope > 0:
        raw_estimate = (model.failure_index - current_index) / fit.slope
    else:
        raw_estimate = float(RUL_CAP)
    estimated_rul = int(round(min(max(raw_estimate, 0.0), float(RUL_CAP))))

    return RulEstimate(
        split=split,
        unit_id=unit_id,
        latest_cycle=cycles[-1],
        cycles_analysed=len(analysed),
        current_index=current_index,
        index_slope_per_cycle=fit.slope,
        raw_estimate=raw_estimate,
        estimated_rul=estimated_rul,
    )
