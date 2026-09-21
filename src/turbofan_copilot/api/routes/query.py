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
from starlette.concurrency import run_in_threadpool

from turbofan_copilot.api.dependencies import QueryService, get_db_session, get_query_service
from turbofan_copilot.api.schemas import QueryRequest, QueryStreamEvent
from turbofan_copilot.api.security import require_api_key
from turbofan_copilot.core.config import Settings
from turbofan_copilot.core.telemetry import QueryTelemetry, collect_query_telemetry, elapsed_ms
from turbofan_copilot.core.tracing import flush_tracing, trace_question
from turbofan_copilot.db.query_log import record_query_run
from turbofan_copilot.llm.answer import NO_EVIDENCE_REASON, NOT_SUPPORTED_REASON, GroundedAnswer
from turbofan_copilot.llm.usage import TokenUsage, estimate_usd

router = APIRouter(prefix="/v1", tags=["query"], dependencies=[Depends(require_api_key)])

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


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


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
            request_id=_request_id(request),
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


# Short, stable codes for the two refusal points, so a log query can group them.
_ABSTENTION_CODES = {NO_EVIDENCE_REASON: "no_evidence", NOT_SUPPORTED_REASON: "not_supported"}


def _log_query_event(
    request: Request,
    *,
    endpoint: str,
    telemetry: QueryTelemetry,
    answer: GroundedAnswer | None,
    latency_ms: int,
) -> None:
    """Write one structured ``query`` event describing how this question went.

    The question text is deliberately left out: it is untrusted input that may
    contain personal data, and ``query_runs`` already stores it under the same
    ``request_id`` for anyone entitled to read it.
    """
    settings = cast(Settings, request.app.state.settings)
    if answer is None:
        outcome = "error"
    elif answer.abstained:
        outcome = "abstained"
    else:
        outcome = "answered"
    usage = TokenUsage(
        calls=telemetry.llm_calls,
        prompt_tokens=telemetry.prompt_tokens,
        completion_tokens=telemetry.completion_tokens,
    )
    health = answer.engine_health if answer is not None else None
    fields: dict[str, object] = {
        "event": "query",
        "request_id": _request_id(request),
        "endpoint": endpoint,
        "outcome": outcome,
        "abstention_reason": (
            _ABSTENTION_CODES.get(answer.abstention_reason or "", "other")
            if answer is not None and answer.abstained
            else None
        ),
        "citation_count": len(answer.citations) if answer is not None else 0,
        "engine_unit_id": health.trend.unit_id if health is not None else None,
        "rul_model": health.rul.model if health is not None else None,
        "latency_ms": latency_ms,
        "stage_ms": telemetry.stage_ms,
        "llm_calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "model": settings.openai_model,
        "cost_usd": estimate_usd(settings.openai_model, usage),
        "error_type": telemetry.error_type,
    }
    _logger.info("query %s in %d ms", outcome, latency_ms, extra={"fields": fields})


def _trace_output(answer: GroundedAnswer) -> dict[str, object]:
    """What the trace records as the question's result."""
    return {
        "answer": answer.answer,
        "abstained": answer.abstained,
        "abstention_reason": answer.abstention_reason,
        "citations": [citation.model_dump() for citation in answer.citations],
    }


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
    try:
        with (
            collect_query_telemetry() as telemetry,
            trace_question(
                _request_id(request), endpoint="query", question=payload.question
            ) as trace,
        ):
            started = perf_counter()
            try:
                answer = service.answer(payload.question)
            except Exception as error:
                telemetry.error_type = type(error).__name__
                trace.update(level="ERROR", status_message=telemetry.error_type)
                _log_query_event(
                    request,
                    endpoint="query",
                    telemetry=telemetry,
                    answer=None,
                    latency_ms=elapsed_ms(started),
                )
                raise
            latency_ms = elapsed_ms(started)
            trace.update(output=_trace_output(answer))
            _persist_run(db, request, payload.question, answer, latency_ms)
            _log_query_event(
                request, endpoint="query", telemetry=telemetry, answer=answer, latency_ms=latency_ms
            )
    finally:
        # After the root span has ended, so the whole trace is sent.
        flush_tracing()
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
        # Collected across the whole stream: the body is produced here, after the
        # route function has already returned the response object.
        try:
            with (
                collect_query_telemetry() as telemetry,
                trace_question(
                    _request_id(request), endpoint="stream", question=payload.question
                ) as trace,
            ):
                started = perf_counter()
                answer: GroundedAnswer | None = None
                async for event in service.answer_events(payload.question):
                    if event.event == "answer":
                        answer = GroundedAnswer.model_validate(event.data)
                    yield _to_sse(event)
                latency_ms = elapsed_ms(started)
                if answer is not None:
                    trace.update(output=_trace_output(answer))
                    _persist_run(db, request, payload.question, answer, latency_ms)
                else:
                    trace.update(level="ERROR", status_message=telemetry.error_type or "no answer")
                _log_query_event(
                    request,
                    endpoint="stream",
                    telemetry=telemetry,
                    answer=answer,
                    latency_ms=latency_ms,
                )
        finally:
            # The client already has the answer; sending the spans must not block
            # the event loop, so it runs in the thread pool.
            await run_in_threadpool(flush_tracing)

    return StreamingResponse(frames(), media_type="text/event-stream")
