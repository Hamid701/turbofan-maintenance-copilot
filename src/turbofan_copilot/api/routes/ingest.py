"""The authenticated ingestion endpoint.

The boundary is the point here: every call must present the configured
``X-API-Key``. The ingestion itself still runs out of band via
``scripts/ingest_*.py`` - a synchronous HTTP request is the wrong place to
re-embed the corpus, and a job runner arrives in Milestone 8. This endpoint
authenticates the request, stamps it with the request ID, and records it.
"""

import logging

from fastapi import APIRouter, Depends, Request

from turbofan_copilot.api.schemas import IngestAccepted, IngestRequest
from turbofan_copilot.api.security import require_ingest_api_key

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])

_logger = logging.getLogger("turbofan_copilot.api.ingest")

_DEFERRED_DETAIL = (
    "Request authenticated and recorded. Ingestion runs out of band via "
    "scripts/ingest_*.py until a job runner exists (Milestone 8)."
)


@router.post(
    "",
    status_code=202,
    response_model=IngestAccepted,
    summary="Request (re)ingestion of a stored dataset (authenticated)",
    dependencies=[Depends(require_ingest_api_key)],
)
def post_ingest(payload: IngestRequest, request: Request) -> IngestAccepted:
    """Accept an authenticated ingestion request for one dataset."""
    request_id = str(getattr(request.state, "request_id", "unknown"))
    _logger.info("ingestion requested: dataset=%s [%s]", payload.dataset, request_id)
    return IngestAccepted(
        dataset=payload.dataset,
        request_id=request_id,
        detail=_DEFERRED_DETAIL,
    )
