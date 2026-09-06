"""Deterministic per-sensor trend summary for one turbofan engine.

Pure calculation: no database, no LLM, no invented physical sensor names.
"""

from statistics import fmean, linear_regression

import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS

DEFAULT_WINDOW = 30
DEFAULT_TOP_MOVERS = 5
_ZERO_MEAN_TOLERANCE = 1e-9


class SensorTrend(BaseModel):
    """One sensor's least-squares trend over the analysed window of cycles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sensor: str
    slope_per_cycle: float
    total_change: float
    window_mean: float
    relative_change: float


class EngineHealthSummary(BaseModel):
    """A deterministic snapshot of one engine's recent sensor behaviour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    unit_id: int
    cycles_observed: int
    cycles_analysed: int
    first_analysed_cycle: int
    latest_cycle: int
    trends: tuple[SensorTrend, ...]
    biggest_movers: tuple[str, ...]


def summarise_engine_health(
    readings: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    top_movers: int = DEFAULT_TOP_MOVERS,
) -> EngineHealthSummary:
    """Summarise the last ``window`` cycles of one engine's sensor readings.

    ``readings`` must hold one engine's rows (one ``split``, one ``unit_id``) with
    a ``cycle`` column and the 21 ``sensor_*`` columns. Fewer cycles than ``window``
    are used as-is; fewer than two raise.
    """
    if window < 2:
        raise ValueError("window must be at least 2 cycles")
    if top_movers < 1:
        raise ValueError("top_movers must be at least 1")

    splits = set(readings["split"])
    units = set(readings["unit_id"])
    if len(splits) != 1 or len(units) != 1:
        raise ValueError("readings must contain exactly one (split, unit_id)")

    ordered = readings.sort_values("cycle")
    cycles_observed = len(ordered)
    if cycles_observed < 2:
        raise ValueError("an engine needs at least two cycles to show a trend")

    analysed = ordered.tail(window)
    cycle_values = [int(value) for value in analysed["cycle"].tolist()]
    span = cycle_values[-1] - cycle_values[0]

    trends: list[SensorTrend] = []
    for sensor in SENSOR_COLUMNS:
        sensor_values = [float(value) for value in analysed[sensor].tolist()]
        slope = linear_regression(cycle_values, sensor_values).slope
        total_change = slope * span
        mean = fmean(sensor_values)
        relative_change = total_change / mean if abs(mean) > _ZERO_MEAN_TOLERANCE else 0.0
        trends.append(
            SensorTrend(
                sensor=sensor,
                slope_per_cycle=slope,
                total_change=total_change,
                window_mean=mean,
                relative_change=relative_change,
            )
        )

    ranked = sorted(trends, key=lambda trend: abs(trend.relative_change), reverse=True)
    biggest_movers = tuple(trend.sensor for trend in ranked[:top_movers])

    return EngineHealthSummary(
        split=str(next(iter(splits))),
        unit_id=int(next(iter(units))),
        cycles_observed=cycles_observed,
        cycles_analysed=len(analysed),
        first_analysed_cycle=cycle_values[0],
        latest_cycle=cycle_values[-1],
        trends=tuple(trends),
        biggest_movers=biggest_movers,
    )
