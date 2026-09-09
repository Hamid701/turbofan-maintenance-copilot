"""Honest evaluation protocol for remaining-useful-life predictors.

Pure calculation: no database, no model. This module owns the three things every
RUL candidate must share - a grouped train/validation split, the piecewise-linear
target, and the error metrics - so that models are compared like for like and the
FD001 test targets stay untouched until a finalist is chosen.
"""

from collections.abc import Callable, Iterable, Sequence
from math import exp, sqrt
from random import Random
from statistics import fmean

import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbofan_copilot.health.degradation import RUL_CAP
from turbofan_copilot.health.engine_health import DEFAULT_WINDOW
from turbofan_copilot.health.rul_features import piecewise_rul

DEFAULT_SEED = 20260908
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_DRAWS_PER_ENGINE = 5
MIN_HISTORY_CYCLES = DEFAULT_WINDOW
MIN_TRUE_RUL = 5
LATE_PENALTY_SCALE = 10.0
EARLY_PENALTY_SCALE = 13.0

type RulCase = tuple[pd.DataFrame, int]
"""One engine's history up to a cut point, and its true remaining life there."""


class RulScore(BaseModel):
    """Error summary for one predictor over one set of cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: int
    mae: float
    rmse: float
    mean_error: float
    nasa_score: float


def split_engine_ids(
    unit_ids: Iterable[int],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split engine ids into (fit, validation), grouped so no engine spans both.

    Rows from one engine are not independent observations, so the split has to be
    made over engines. It is seeded, so the same ids always produce the same split.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = sorted({int(unit) for unit in unit_ids})
    if len(ordered) < 2:
        raise ValueError("at least two engines are needed to split")

    held_out = max(1, round(len(ordered) * validation_fraction))
    if held_out >= len(ordered):
        raise ValueError("validation_fraction leaves no engines to fit on")

    shuffled = list(ordered)
    Random(seed).shuffle(shuffled)
    validation = tuple(sorted(shuffled[:held_out]))
    fit = tuple(sorted(shuffled[held_out:]))
    return fit, validation


def validation_cases(
    train_readings: pd.DataFrame,
    unit_ids: Sequence[int],
    *,
    draws_per_engine: int = DEFAULT_DRAWS_PER_ENGINE,
    seed: int = DEFAULT_SEED,
    cap: int = RUL_CAP,
) -> tuple[RulCase, ...]:
    """Truncate held-out run-to-failure engines into test-set-shaped cases.

    The FD001 test engines were cut at a random point before failure, so a held-out
    train engine only resembles one after the same treatment. Each engine is cut
    ``draws_per_engine`` times at seeded random remaining lives, which keeps the
    case count near the test set's 100 without needing 100 held-out engines.
    """
    if draws_per_engine < 1:
        raise ValueError("draws_per_engine must be at least 1")

    rng = Random(seed)
    cases: list[RulCase] = []
    for unit in sorted(int(unit) for unit in unit_ids):
        engine = train_readings.loc[train_readings["unit_id"] == unit].sort_values("cycle")
        if engine.empty:
            raise LookupError(f"no readings for engine {unit}")
        final_cycle = int(engine["cycle"].max())

        highest_draw = min(cap, final_cycle - MIN_HISTORY_CYCLES)
        if highest_draw < MIN_TRUE_RUL:
            continue
        for _ in range(draws_per_engine):
            remaining = rng.randint(MIN_TRUE_RUL, highest_draw)
            history = engine.loc[engine["cycle"] <= final_cycle - remaining]
            cases.append((history, piecewise_rul(remaining, cap=cap)))
    if not cases:
        raise ValueError("no engine was long enough to produce a validation case")
    return tuple(cases)


def nasa_score(errors: Iterable[float]) -> float:
    """Return the C-MAPSS asymmetric score; lower is better.

    A late prediction (the engine failed before the estimate said it would) is
    penalised harder than an equally sized early one. The score is a sum, so it is
    only comparable between runs with the same number of cases.
    """
    return sum(
        exp(error / LATE_PENALTY_SCALE) - 1.0
        if error >= 0
        else exp(-error / EARLY_PENALTY_SCALE) - 1.0
        for error in errors
    )


def score_rul(predictions: Sequence[float], truths: Sequence[int]) -> RulScore:
    """Return MAE, RMSE, bias, and the NASA score for one predictor's output."""
    if len(predictions) != len(truths):
        raise ValueError("predictions and truths must be the same length")
    if not predictions:
        raise ValueError("scoring needs at least one case")

    errors = [
        float(prediction) - float(truth)
        for prediction, truth in zip(predictions, truths, strict=True)
    ]
    return RulScore(
        cases=len(errors),
        mae=round(fmean(abs(error) for error in errors), 2),
        rmse=round(sqrt(fmean(error * error for error in errors)), 2),
        mean_error=round(fmean(errors), 2),
        nasa_score=round(nasa_score(errors), 1),
    )


def evaluate_predictor(
    cases: Iterable[RulCase],
    predict: Callable[[pd.DataFrame], float],
) -> RulScore:
    """Score any callable that turns one engine's history into an estimate.

    Typed on a plain callable so a deterministic baseline, a fitted regressor, and
    a foundation model can all be scored by the same protocol.
    """
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("evaluation needs at least one case")
    predictions = [predict(history) for history, _ in case_list]
    return score_rul(predictions, [truth for _, truth in case_list])
