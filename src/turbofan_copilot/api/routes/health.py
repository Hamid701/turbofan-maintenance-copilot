"""Liveness and readiness endpoints."""

from typing import cast

from fastapi import APIRouter, Request, Response, status

from turbofan_copilot.api.readiness import run_readiness_checks
from turbofan_copilot.api.schemas import HealthResponse, ReadinessResponse
from turbofan_copilot.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Check API liveness")
def get_health(request: Request) -> HealthResponse:
    """Confirm that the API and its validated configuration are available."""
    settings = cast(Settings, request.app.state.settings)
    return HealthResponse(
        service=settings.app_name,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Check whether this instance can serve queries",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "At least one dependency is unavailable.",
            "model": ReadinessResponse,
        }
    },
)
def get_readiness(request: Request, response: Response) -> ReadinessResponse:
    """Report every dependency check, and answer 503 when any of them fails.

    Separate from ``/health`` on purpose. A failed liveness probe means restart
    this container; a failed readiness probe means stop sending it traffic. Using
    one endpoint for both turns a brief database outage into a restart loop.

    This does not build the query pipeline - that takes about twenty seconds and
    would time out any sane probe. It reports that the pipeline's dependencies are
    available, not that it is already warm.
    """
    settings = cast(Settings, request.app.state.settings)
    checks = run_readiness_checks(settings)
    ready = all(check.ok for check in checks)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
