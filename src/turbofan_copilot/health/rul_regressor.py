"""Fitted RUL models over the windowed sensor features: ridge, then boosted trees.

Ridge is the sanity check for the representation: if a straight line through 61
features cannot beat a constant, the features carry no signal. It uses L2
regularization because the features are strongly correlated with each other - a
sensor's last value, window mean, and first-to-last change all describe the same
drift - and an unpenalised fit answers that by splitting a huge positive weight
against a huge negative one. ``alpha`` is the strength of that penalty.

Gradient boosting is the next question: is the relationship between wear and
remaining life *curved*? Ridge can only add the features up. Boosting fits a small
tree to the current errors, adds it to the running total, and repeats, so it can
represent "this sensor only matters once it has drifted past here" and the way
C-MAPSS wear accelerates near failure. ``learning_rate`` scales each tree's
contribution, ``max_leaf_nodes`` caps how detailed one tree may be, and ``max_iter``
is how many trees are added.

Both return the same ``RulRegressor``, so every model is scored through one contract.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from turbofan_copilot.health.degradation import RUL_CAP
from turbofan_copilot.health.rul_features import (
    FEATURE_WINDOW,
    informative_sensors,
    training_table,
    window_features,
)

DEFAULT_ALPHA = 10.0
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_LEAF_NODES = 15
DEFAULT_MAX_ITER = 400
RANDOM_STATE = 0

# Mean absolute error of the chosen configuration over the 100 held-out FD001 test
# engines (ADR-001, `scripts/evaluate_rul.py --test`). Reported alongside every
# prediction so a reader can size the estimate instead of reading it as exact.
BOOSTED_TEST_MAE = 8.4


class RulRegressor:
    """A fitted estimator plus the sensor list and window it was fitted with."""

    def __init__(self, estimator: Any, sensors: Sequence[str], *, window: int) -> None:
        self.estimator = estimator
        self.sensors = tuple(sensors)
        self.window = window

    def predict(self, history: pd.DataFrame) -> float:
        """Estimate one engine's remaining cycles from its recent readings.

        Clipped to ``[0, RUL_CAP]`` and rounded like the deterministic baseline, so
        every model is scored on exactly the same output contract.
        """
        features = window_features(history, self.sensors, window=self.window)
        row = pd.DataFrame([features], columns=list(self.feature_columns))
        raw = float(self.estimator.predict(row)[0])
        return float(round(min(max(raw, 0.0), float(RUL_CAP))))

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """The column names the estimator was fitted on, in order."""
        return tuple(self.estimator.feature_names_in_)


def _fit(estimator: Any, labelled_readings: pd.DataFrame, window: int) -> RulRegressor:
    """Select the informative sensors, build the training table, and fit."""
    sensors = informative_sensors(labelled_readings)
    features, targets = training_table(labelled_readings, sensors, window=window)
    estimator.fit(features, targets)
    return RulRegressor(estimator, sensors, window=window)


def fit_rul_regressor(
    labelled_readings: pd.DataFrame,
    *,
    alpha: float = DEFAULT_ALPHA,
    window: int = FEATURE_WINDOW,
) -> RulRegressor:
    """Fit a scaler and ridge on every full window in the labelled train engines.

    The scaler is inside the pipeline on purpose: it learns its means and standard
    deviations from the fitting rows only, so nothing about the held-out engines can
    leak into the model through the normalization.
    """
    return _fit(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), labelled_readings, window)


def fit_boosted_rul_regressor(
    labelled_readings: pd.DataFrame,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_leaf_nodes: int = DEFAULT_MAX_LEAF_NODES,
    max_iter: int = DEFAULT_MAX_ITER,
    window: int = FEATURE_WINDOW,
) -> RulRegressor:
    """Fit gradient-boosted trees on the same features. No scaler - trees ignore scale.

    ``early_stopping`` is switched off deliberately. scikit-learn would otherwise
    hold out a slice of the training rows at random to decide when to stop, and rows
    from one engine are not independent - the same engine would land on both sides of
    that internal split and make stopping look better than it is. Complexity is
    controlled by choosing ``max_iter`` on the engine-grouped validation split instead.
    ``random_state`` is fixed so the same data always produces the same model.
    """
    estimator = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return _fit(estimator, labelled_readings, window)
