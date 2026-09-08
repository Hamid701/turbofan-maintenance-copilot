"""Turn an engine reference into a deterministic health report the pipeline can use."""

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import load_engine_readings, load_split_readings
from turbofan_copilot.health.degradation import (
    DegradationModel,
    RulEstimate,
    estimate_engine_rul,
    fit_degradation_model,
)
from turbofan_copilot.health.engine_health import EngineHealthSummary, summarise_engine_health
from turbofan_copilot.llm.router import EngineReference


class EngineHealthReport(BaseModel):
    """Both deterministic engine-health tool outputs for one engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trend: EngineHealthSummary
    rul: RulEstimate


def load_degradation_model(session: Session) -> DegradationModel:
    """Fit the degradation model once from the stored train set."""
    return fit_degradation_model(load_split_readings(session, "train"))


def build_engine_health_report(
    session: Session,
    reference: EngineReference,
    *,
    degradation_model: DegradationModel,
) -> EngineHealthReport:
    """Run both deterministic health tools for the referenced engine."""
    readings = load_engine_readings(session, reference.split, reference.unit_id)
    return EngineHealthReport(
        trend=summarise_engine_health(readings),
        rul=estimate_engine_rul(readings, degradation_model),
    )
