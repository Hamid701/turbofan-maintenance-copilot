"""The feedback endpoint.

A reader tells us whether an earlier answer was useful, quoting the
``X-Request-ID`` that answer carried. The submission is validated and written to
the ``feedback`` table; here the write *is* the point, so a failure is a 500.
"""

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from turbofan_copilot.api.dependencies import get_db_session
from turbofan_copilot.api.schemas import FeedbackAccepted, FeedbackRequest
from turbofan_copilot.db.query_log import record_feedback

router = APIRouter(prefix="/v1", tags=["feedback"])

_logger = logging.getLogger("turbofan_copilot.api.feedback")


@router.post(
    "/feedback",
    status_code=202,
    response_model=FeedbackAccepted,
    summary="Record a reader's verdict on an earlier answer",
)
def post_feedback(
    payload: FeedbackRequest,
    request: Request,
    db: Session = Depends(get_db_session),
) -> FeedbackAccepted:
    """Persist one feedback submission and acknowledge it."""
    request_id = str(getattr(request.state, "request_id", "unknown"))
    record_feedback(
        db,
        query_request_id=payload.query_request_id,
        rating=payload.rating,
        comment=payload.comment,
    )
    db.commit()
    _logger.info(
        "feedback recorded: query=%s rating=%s [%s]",
        payload.query_request_id,
        payload.rating,
        request_id,
    )
    return FeedbackAccepted(
        query_request_id=payload.query_request_id,
        rating=payload.rating,
        request_id=request_id,
    )
