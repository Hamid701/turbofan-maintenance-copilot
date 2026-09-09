"""Turn an engine reference into a deterministic health report the pipeline can use."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import load_engine_readings, load_split_readings
from turbofan_copilot.health.degradation import (
    BASELINE_TEST_MAE,
    DegradationModel,
    estimate_engine_rul,
    fit_degradation_model,
)
from turbofan_copilot.health.engine_health import EngineHealthSummary, summarise_engine_health
from turbofan_copilot.health.rul_features import label_train_readings
from turbofan_copilot.health.rul_regressor import (
    BOOSTED_TEST_MAE,
    RulRegressor,
    fit_boosted_rul_regressor,
)
from turbofan_copilot.llm.router import EngineReference

type RulModelName = Literal["boosted-trees", "degradation-index"]


class RulPrediction(BaseModel):
    """One engine's estimated remaining life, and how much to trust it.

    The estimate travels with the name of the model that produced it and that
    model's measured error, because a maintenance answer that states a bare number
    invites it to be read as a measurement. It is a prediction, and the safety
    boundaries require the difference to be visible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimated_rul: int
    model: RulModelName
    typical_error_cycles: float
    cycles_observed: int


class EngineHealthReport(BaseModel):
    """Both deterministic engine-health tool outputs for one engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend: EngineHealthSummary
    rul: RulPrediction


def load_degradation_model(session: Session) -> DegradationModel:
    """Fit the degradation model once from the stored train set."""
    return fit_degradation_model(load_split_readings(session, "train"))


def load_rul_regressor(session: Session) -> RulRegressor:
    """Fit the chosen RUL model once from the stored train set (ADR-001).

    Refitting beats persisting a serialised model here: it takes seconds over data
    that already has to be in the database, and it removes any question of an
    artifact drifting out of step with the rows it was trained on.
    """
    return fit_boosted_rul_regressor(label_train_readings(load_split_readings(session, "train")))


def predict_rul(
    readings: pd.DataFrame,
    *,
    regressor: RulRegressor,
    degradation_model: DegradationModel,
) -> RulPrediction:
    """Estimate remaining life, falling back when the history is too short.

    The learned model needs a full feature window and refuses anything shorter,
    so a young engine would otherwise get an error instead of an answer. The
    deterministic baseline handles short histories by design, so it covers that
    case and the report says which model replied.
    """
    cycles_observed = len(readings)
    if cycles_observed >= regressor.window:
        return RulPrediction(
            estimated_rul=int(regressor.predict(readings)),
            model="boosted-trees",
            typical_error_cycles=BOOSTED_TEST_MAE,
            cycles_observed=cycles_observed,
        )
    return RulPrediction(
        estimated_rul=estimate_engine_rul(readings, degradation_model).estimated_rul,
        model="degradation-index",
        typical_error_cycles=BASELINE_TEST_MAE,
        cycles_observed=cycles_observed,
    )


def build_engine_health_report(
    session: Session,
    reference: EngineReference,
    *,
    degradation_model: DegradationModel,
    rul_regressor: RulRegressor,
) -> EngineHealthReport:
    """Run both deterministic health tools for the referenced engine."""
    readings = load_engine_readings(session, reference.split, reference.unit_id)
    return EngineHealthReport(
        trend=summarise_engine_health(readings),
        rul=predict_rul(readings, regressor=rul_regressor, degradation_model=degradation_model),
    )
