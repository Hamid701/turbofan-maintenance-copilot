"""The authentication boundary for the ingestion endpoints.

Ingestion changes stored data, so it is gated by a shared secret sent in the
``X-API-Key`` header and compared in constant time. If no key is configured the
endpoints answer ``503`` - they are never left open.
"""

import secrets
from typing import cast

from fastapi import Request
from starlette.exceptions import HTTPException

from turbofan_copilot.core.config import Settings

API_KEY_HEADER = "X-API-Key"


def require_ingest_api_key(request: Request) -> None:
    """Allow the request only if it carries the configured ingestion key."""
    settings = cast(Settings, request.app.state.settings)
    configured = settings.ingest_api_key
    if configured is None:
        raise HTTPException(status_code=503, detail="Ingestion is not configured.")

    supplied = request.headers.get(API_KEY_HEADER, "")
    if not secrets.compare_digest(supplied, configured.get_secret_value()):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")
