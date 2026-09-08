"""The grounded-answer query endpoints: one blocking, one streaming.

Both record a ``query_runs`` row after answering. The write is best-effort - a
logging failure is logged and swallowed so the caller still gets their answer.
"""

import json
import logging
from collections.abc import AsyncIterator
from time import perf_counter
from typing import cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from turbofan_copilot.api.dependencies import QueryService, get_db_session, get_query_service
from turbofan_copilot.api.schemas import QueryRequest, QueryStreamEvent
from turbofan_copilot.core.config import Settings
from turbofan_copilot.db.query_log import record_query_run
from turbofan_copilot.llm.answer import GroundedAnswer

router = APIRouter(prefix="/v1", tags=["query"])

_logger = logging.getLogger("turbofan_copilot.api.query")

_ANSWER_EXAMPLE = {
    "answer": (
        "A magnetic chip detector sits in the oil system and captures ferrous debris. "
        "Metal on the probe points to internal wear and warrants further inspection."
    ),
    "citations": [
        {
            "source_id": "faa-h-8083-32b-chapter-6",
            "pdf_page_number": 28,
            "printed_page_label": "6-28",
        }
    ],
    "engine_health": None,
    "abstained": False,
    "abstention_reason": None,
}


def _persist_run(
    db: Session,
    request: Request,
    question: str,
    answer: GroundedAnswer,
    latency_ms: int,
) -> None:
    """Record one answered query; never raise into the request path."""
    settings = cast(Settings, request.app.state.settings)
    engine_unit_id = (
        answer.engine_health.trend.unit_id if answer.engine_health is not None else None
    )
    try:
        record_query_run(
            db,
            request_id=str(getattr(request.state, "request_id", "unknown")),
            question=question,
            abstained=answer.abstained,
            citations=[citation.model_dump() for citation in answer.citations],
            engine_unit_id=engine_unit_id,
            answer_model=settings.openai_model,
            latency_ms=latency_ms,
        )
        db.commit()
    except Exception:
        _logger.exception("failed to record query run")
        db.rollback()


@router.post(
    "/query",
    response_model=GroundedAnswer,
    summary="Answer a turbine-maintenance question with manual citations",
    responses={200: {"content": {"application/json": {"example": _ANSWER_EXAMPLE}}}},
)
def post_query(
    payload: QueryRequest,
    request: Request,
    service: QueryService = Depends(get_query_service),
    db: Session = Depends(get_db_session),
) -> GroundedAnswer:
    """Run the retrieval-and-answer pipeline, or return a structured abstention."""
    started = perf_counter()
    answer = service.answer(payload.question)
    _persist_run(db, request, payload.question, answer, round((perf_counter() - started) * 1000))
    return answer


def _to_sse(event: QueryStreamEvent) -> str:
    """Render one event in the Server-Sent Events wire format."""
    return f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"


@router.post(
    "/query/stream",
    summary="Answer a question, streaming a progress marker per pipeline stage",
    response_class=StreamingResponse,
)
async def post_query_stream(
    payload: QueryRequest,
    request: Request,
    service: QueryService = Depends(get_query_service),
    db: Session = Depends(get_db_session),
) -> StreamingResponse:
    """Stream ``routed`` / ``retrieved`` / ``generating`` markers, then ``answer``.

    The ``answer`` event carries the same ``GroundedAnswer`` as ``POST /v1/query``;
    an ``error`` event ends the stream if the pipeline fails after it has begun.
    """

    async def frames() -> AsyncIterator[str]:
        started = perf_counter()
        answer: GroundedAnswer | None = None
        async for event in service.answer_events(payload.question):
            if event.event == "answer":
                answer = GroundedAnswer.model_validate(event.data)
            yield _to_sse(event)
        if answer is not None:
            _persist_run(
                db, request, payload.question, answer, round((perf_counter() - started) * 1000)
            )

    return StreamingResponse(frames(), media_type="text/event-stream")
