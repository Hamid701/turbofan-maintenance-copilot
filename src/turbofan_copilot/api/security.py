"""The API-key boundary for the ``/v1`` endpoints.

Every question spends the owner's OpenAI key and every feedback submission writes
a row, so both are gated by a shared secret sent in the ``X-API-Key`` header. If
no key is configured the endpoints answer ``503``: they fail closed, never open,
because a deployment that forgot one variable should break loudly rather than
quietly run up a bill.
"""

import secrets
from typing import cast

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException

from turbofan_copilot.core.config import Settings

API_KEY_HEADER = "X-API-Key"

# Declared through FastAPI's security helper so /docs shows an Authorize button.
# auto_error is off so this module, not the helper, decides between 401 and 503.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_api_key(
    request: Request,
    supplied: str | None = Depends(_api_key_header),
) -> None:
    """Allow the request only if it carries the configured API key."""
    configured = cast(Settings, request.app.state.settings).query_api_key
    if configured is None:
        raise HTTPException(status_code=503, detail="The API key is not configured.")

    # Compared as bytes: compare_digest rejects non-ASCII str with a TypeError, and
    # Starlette decodes header bytes as Latin-1, so one odd byte would become a 500.
    if supplied is None or not secrets.compare_digest(
        supplied.encode(), configured.get_secret_value().encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="A valid X-API-Key header is required.",
            headers={"WWW-Authenticate": "APIKey"},
        )
