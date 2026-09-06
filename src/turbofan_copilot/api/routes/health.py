"""Liveness endpoint."""

from typing import cast

from fastapi import APIRouter, Request

from turbofan_copilot.api.schemas import HealthResponse
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
