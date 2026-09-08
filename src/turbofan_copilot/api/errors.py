"""Exception handlers that give every error a consistent, correlated JSON body."""

import logging
import uuid

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.api.schemas import ErrorResponse

_logger = logging.getLogger("turbofan_copilot.api.error")


def _request_id(request: Request) -> str:
    supplied = getattr(request.state, "request_id", None)
    return supplied if isinstance(supplied, str) else uuid.uuid4().hex


def _json_error(request_id: str, status_code: int, content: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={REQUEST_ID_HEADER: request_id},
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> Response:
    """Turn any unhandled exception into a 500 that carries the request ID."""
    request_id = _request_id(request)
    _logger.exception("unhandled error [%s]", request_id)
    body = ErrorResponse(detail="Internal server error.", request_id=request_id)
    return _json_error(request_id, 500, body.model_dump())


async def handle_http_exception(request: Request, exc: Exception) -> Response:
    """Render a deliberate ``HTTPException`` in the standard error shape."""
    if not isinstance(exc, HTTPException):  # pragma: no cover - registered per type
        return await handle_unexpected_error(request, exc)
    request_id = _request_id(request)
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    body = ErrorResponse(detail=detail, request_id=request_id)
    return _json_error(request_id, exc.status_code, body.model_dump())


async def handle_validation_error(request: Request, exc: Exception) -> Response:
    """Keep FastAPI's field-level 422 detail, add the request ID beside it."""
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - per type
        return await handle_unexpected_error(request, exc)
    request_id = _request_id(request)
    return _json_error(
        request_id,
        422,
        {"detail": jsonable_encoder(exc.errors()), "request_id": request_id},
    )
