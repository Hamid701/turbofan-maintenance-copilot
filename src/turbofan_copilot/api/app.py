"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from turbofan_copilot.api.errors import (
    handle_http_exception,
    handle_unexpected_error,
    handle_validation_error,
)
from turbofan_copilot.api.middleware import request_id_middleware
from turbofan_copilot.api.routes.feedback import router as feedback_router
from turbofan_copilot.api.routes.health import router as health_router
from turbofan_copilot.api.routes.ingest import router as ingest_router
from turbofan_copilot.api.routes.query import router as query_router
from turbofan_copilot.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an API instance around one immutable configuration snapshot."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
    )
    app.state.settings = resolved_settings

    app.middleware("http")(request_id_middleware)
    app.add_exception_handler(Exception, handle_unexpected_error)
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(feedback_router)
    return app
