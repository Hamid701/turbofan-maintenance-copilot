"""FastAPI application factory."""

from fastapi import FastAPI

from turbofan_copilot.api.routes.health import router as health_router
from turbofan_copilot.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an API instance around one immutable configuration snapshot."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
    )
    app.state.settings = resolved_settings
    app.include_router(health_router)
    return app
