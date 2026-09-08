"""Give every request a correlation ID, on the response header and in the log."""

import logging
import uuid
from collections.abc import Awaitable, Callable

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
    """Attach the request ID to ``request.state``, the response, and one log line."""
    request_id = _resolve_request_id(request)
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        _logger.exception(
            "%s %s -> unhandled error [%s]", request.method, request.url.path, request_id
        )
        raise

    response.headers[REQUEST_ID_HEADER] = request_id
    _logger.info(
        "%s %s -> %d [%s]",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
    )
    return response
