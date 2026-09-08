"""Public request and response schemas for the API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from turbofan_copilot.core.config import RuntimeEnvironment


class HealthResponse(BaseModel):
    """Liveness information that is safe to expose without authentication."""

    status: Literal["ok"] = "ok"
    service: str
    environment: RuntimeEnvironment


class ErrorResponse(BaseModel):
    """The body returned for a deliberate HTTP error or an unhandled exception.

    ``request_id`` matches the ``X-Request-ID`` response header so a user report
    can be tied to a log line. Request-validation errors (422) keep FastAPI's
    field-level ``detail`` list and add ``request_id`` alongside it.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str
    request_id: str


class QueryRequest(BaseModel):
    """A maintenance question for the grounded-answer pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "How does a magnetic chip detector indicate internal engine wear?"},
                {"question": "Engine 5 shows rising EGT during start - what should be inspected?"},
            ]
        },
    )

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    ]


IngestDataset = Literal["corpus", "fd001"]


class IngestRequest(BaseModel):
    """Which stored dataset an authenticated caller wants (re)ingested."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"dataset": "corpus"}]},
    )

    dataset: IngestDataset


class IngestAccepted(BaseModel):
    """Acknowledgement that an authenticated ingestion request was accepted."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted"] = "accepted"
    dataset: IngestDataset
    request_id: str
    detail: str


FeedbackRating = Literal["up", "down"]


class FeedbackRequest(BaseModel):
    """A reader's verdict on one earlier answer, keyed by that answer's request ID."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query_request_id": "3f9c1b2a4d5e6f708192a3b4c5d6e7f8",
                    "rating": "up",
                    "comment": "Pointed me straight to the right manual section.",
                }
            ]
        },
    )

    query_request_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    rating: FeedbackRating
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] | None = None


class FeedbackAccepted(BaseModel):
    """Acknowledgement that a feedback submission was recorded."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted"] = "accepted"
    query_request_id: str
    rating: FeedbackRating
    request_id: str


QueryStreamEventName = Literal["routed", "retrieved", "generating", "answer", "error"]


class QueryStreamEvent(BaseModel):
    """One Server-Sent Event from the streaming query endpoint.

    ``routed`` / ``retrieved`` / ``generating`` are progress markers with an empty
    ``data``. ``answer`` carries the full ``GroundedAnswer`` as JSON. ``error``
    carries a ``detail`` string and ends the stream.
    """

    model_config = ConfigDict(extra="forbid")

    event: QueryStreamEventName
    data: dict[str, object] = Field(default_factory=dict)
