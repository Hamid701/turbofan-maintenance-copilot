"""Write helpers for the served-query log and the feedback table.

Kept apart from the endpoint code so the routes stay thin, and taking primitives
so this module does not depend on the LLM or answer types.
"""

from collections.abc import Sequence

from sqlalchemy.orm import Session

from turbofan_copilot.db.models import Feedback, QueryRun


def record_query_run(
    session: Session,
    *,
    request_id: str,
    question: str,
    abstained: bool,
    citations: Sequence[dict[str, object]],
    engine_unit_id: int | None = None,
    answer_model: str | None = None,
    latency_ms: int | None = None,
) -> QueryRun:
    """Add (not commit) a row describing one answered query."""
    run = QueryRun(
        request_id=request_id,
        question=question,
        abstained=abstained,
        citation_count=len(citations),
        citations=list(citations),
        engine_unit_id=engine_unit_id,
        answer_model=answer_model,
        latency_ms=latency_ms,
    )
    session.add(run)
    return run


def record_feedback(
    session: Session,
    *,
    query_request_id: str,
    rating: str,
    comment: str | None = None,
) -> Feedback:
    """Add (not commit) a feedback row for an earlier answer."""
    entry = Feedback(
        query_request_id=query_request_id,
        rating=rating,
        comment=comment,
    )
    session.add(entry)
    return entry
