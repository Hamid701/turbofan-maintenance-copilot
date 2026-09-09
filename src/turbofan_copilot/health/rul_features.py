"""Windowed features: one engine's recent cycles become one fixed-length row.

Pure calculation: pandas and numpy only, no model and no database. The degradation
baseline compressed 21 sensors into a single wear index; this keeps each sensor
separate and describes its recent behaviour four ways, so a fitted model can decide
for itself which signals matter.

``features_from_window`` is the one definition of a feature row. Both the training
table and the prediction path call it, so the features a model is fitted on cannot
drift away from the features it is later asked to predict from.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from turbofan_copilot.health.degradation import RUL_CAP, STD_FLOOR
from turbofan_copilot.health.engine_health import DEFAULT_WINDOW
from turbofan_copilot.ingestion.fd001 import SENSOR_COLUMNS

FEATURE_WINDOW = DEFAULT_WINDOW
CYCLES_RUN = "cycles_run"


def piecewise_rul(remaining_cycles: int, *, cap: int = RUL_CAP) -> int:
    """Return the capped remaining life used as the supervised target.

    An engine early in its life shows no measurable degradation, so asking a model
    to predict its true distance to failure teaches it noise. The C-MAPSS
    convention is to hold the target flat at the cap until wear becomes visible.
    """
    if remaining_cycles < 0:
        raise ValueError("remaining_cycles must not be negative")
    return min(remaining_cycles, cap)


def label_train_readings(train_readings: pd.DataFrame, *, cap: int = RUL_CAP) -> pd.DataFrame:
    """Return the run-to-failure frame with a piecewise-linear ``rul`` column.

    Every train engine ends at its failure, so remaining life at any row is that
    engine's last cycle minus this cycle, held at ``cap``.
    """
    if "rul" in train_readings.columns:
        raise ValueError("train_readings already carries a rul column")
    labelled = train_readings.copy()
    final_cycle = labelled.groupby("unit_id")["cycle"].transform("max")
    remaining = final_cycle - labelled["cycle"]
    labelled["rul"] = remaining.clip(upper=cap).astype("int64")
    return labelled


def informative_sensors(train_readings: pd.DataFrame) -> tuple[str, ...]:
    """Return the sensors that actually vary across the training data.

    Same variance floor the degradation baseline uses: a sensor that never moves
    carries no information, and dropping it by measurement beats hand-picking.
    """
    varying = tuple(
        name for name in SENSOR_COLUMNS if float(train_readings[name].std(ddof=0)) > STD_FLOOR
    )
    if not varying:
        raise ValueError("no informative sensors found in the training data")
    return varying


def feature_names(sensors: Sequence[str]) -> tuple[str, ...]:
    """Return the feature-row column names, in the order the values are built."""
    suffixes = ("last", "mean", "slope", "delta")
    return (CYCLES_RUN, *(f"{sensor}_{suffix}" for sensor in sensors for suffix in suffixes))


def features_from_window(
    window_values: np.ndarray,
    sensors: Sequence[str],
    cycle: int,
) -> list[float]:
    """Describe one window of sensor readings as a flat list of numbers.

    ``window_values`` is (cycles, sensors) with the most recent cycle last. Each
    sensor contributes its latest value, its window mean, its least-squares slope
    per cycle, and its first-to-last change. ``cycle`` - how long the engine has
    been running - is included because age is itself evidence of wear.

    The slope is a dot product rather than a line fit: for evenly spaced cycles the
    least-squares slope is ``sum((x - mean(x)) * y) / sum((x - mean(x)) ** 2)``, and
    the x term is the same for every window, so it is computed once.
    """
    rows = window_values.shape[0]
    if rows < 2:
        raise ValueError("a feature window needs at least two cycles")
    if window_values.shape[1] != len(sensors):
        raise ValueError("window_values columns must match the sensor list")

    offsets = np.arange(rows, dtype=float)
    centred = offsets - offsets.mean()
    slopes = (centred @ window_values) / float((centred**2).sum())

    values: list[float] = [float(cycle)]
    for index in range(len(sensors)):
        column = window_values[:, index]
        values.extend(
            (
                float(column[-1]),
                float(column.mean()),
                float(slopes[index]),
                float(column[-1] - column[0]),
            )
        )
    return values


def window_features(
    history: pd.DataFrame,
    sensors: Sequence[str],
    *,
    window: int = FEATURE_WINDOW,
) -> list[float]:
    """Build one feature row from the most recent cycles of one engine's history.

    A history shorter than the window is refused rather than shortened. Every
    training example the model ever saw came from a full window, so a partial one
    would silently shift the mean, slope, and change features onto a different scale
    - a wrong answer that looks like a right one. Refusing lets the caller say "not
    enough history yet", which is a true statement about the engine.
    """
    if window < 2:
        raise ValueError("window must be at least 2 cycles")
    ordered = history.sort_values("cycle")
    if len(ordered) < window:
        raise ValueError(
            f"an engine needs at least {window} cycles for a feature row; got {len(ordered)}"
        )

    recent = ordered.tail(window)
    return features_from_window(
        recent[list(sensors)].to_numpy(dtype=float),
        sensors,
        int(recent["cycle"].iloc[-1]),
    )


def training_table(
    labelled_readings: pd.DataFrame,
    sensors: Sequence[str],
    *,
    window: int = FEATURE_WINDOW,
) -> tuple[pd.DataFrame, pd.Series]:
    """Turn labelled run-to-failure engines into a feature matrix and its targets.

    Every cycle with a full window behind it becomes one training example, which is
    what makes ~20k rows out of 100 engines. Cycles before the first full window are
    skipped rather than padded, so no example is built from invented history.
    """
    if "rul" not in labelled_readings.columns:
        raise ValueError("labelled_readings needs a rul column; call label_train_readings first")
    if window < 2:
        raise ValueError("window must be at least 2 cycles")

    columns = list(sensors)
    rows: list[list[float]] = []
    targets: list[int] = []
    for _, engine in labelled_readings.groupby("unit_id", sort=True):
        ordered = engine.sort_values("cycle")
        values = ordered[columns].to_numpy(dtype=float)
        cycles = ordered["cycle"].to_numpy(dtype=int)
        labels = ordered["rul"].to_numpy(dtype=int)
        for end in range(window, len(ordered) + 1):
            rows.append(
                features_from_window(values[end - window : end], sensors, int(cycles[end - 1]))
            )
            targets.append(int(labels[end - 1]))

    if not rows:
        raise ValueError("no engine was long enough to produce a training example")
    return (
        pd.DataFrame(rows, columns=list(feature_names(sensors))),
        pd.Series(targets, name="rul"),
    )
