"""Give every request a correlation ID, on the response header and in the log."""

import logging
import uuid
from collections.abc import Awaitable, Callable
from time import perf_counter

from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
_MAX_SUPPLIED_LENGTH = 128

_logger = logging.getLogger("turbofan_copilot.api.request")


def _resolve_request_id(request: Request) -> str:
    """Use the client's ``X-Request-ID`` if it is sane, otherwise mint one."""
    supplied = request.headers.get(REQUEST_ID_HEADER, "").strip()
    if supplied and len(supplied) <= _MAX_SUPPLIED_LENGTH and supplied.isprintable():
        return supplied
    return uuid.uuid4().hex


async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach the request ID to ``request.state``, the response, and one log line.

    The line carries its details as structured fields, so a log query can filter
    on ``status`` or ``path`` and chart ``latency_ms`` instead of parsing text.
    ``latency_ms`` is time to the response headers: for a streaming response the
    body is still being sent when this line is written.
    """
    request_id = _resolve_request_id(request)
    request.state.request_id = request_id
    fields: dict[str, object] = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
    }
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        fields["latency_ms"] = round((perf_counter() - started) * 1000)
        _logger.exception(
            "%s %s -> unhandled error", request.method, request.url.path, extra={"fields": fields}
        )
        raise

    response.headers[REQUEST_ID_HEADER] = request_id
    fields["status"] = response.status_code
    fields["latency_ms"] = round((perf_counter() - started) * 1000)
    _logger.info(
        "%s %s -> %d",
        request.method,
        request.url.path,
        response.status_code,
        extra={"fields": fields},
    )
    return response
