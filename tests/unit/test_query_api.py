"""Contract tests for the query endpoints with a fake service (no model, no DB)."""

import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import get_db_session, get_query_service
from turbofan_copilot.api.schemas import QueryStreamEvent
from turbofan_copilot.core.config import RuntimeEnvironment, Settings
from turbofan_copilot.db.models import QueryRun
from turbofan_copilot.llm.answer import Citation, GroundedAnswer, abstention


class FakeSession:
    """A stand-in for a SQLAlchemy session that records adds and commits."""

    def __init__(self, *, fail_on_commit: bool = False) -> None:
        self.added: list[object] = []
        self.committed = False
        self.rolled_back = False
        self._fail_on_commit = fail_on_commit

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        if self._fail_on_commit:
            raise RuntimeError("simulated database failure")
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class FakeQueryService:
    """Return a fixed answer (and a fixed event sequence) for any question."""

    def __init__(
        self,
        answer: GroundedAnswer,
        *,
        events: list[QueryStreamEvent] | None = None,
    ) -> None:
        self._answer = answer
        self._events = events
        self.questions: list[str] = []

    def answer(self, question: str) -> GroundedAnswer:
        self.questions.append(question)
        return self._answer

    async def answer_events(self, question: str) -> AsyncIterator[QueryStreamEvent]:
        self.questions.append(question)
        events = self._events
        if events is None:
            events = [
                QueryStreamEvent(event="routed"),
                QueryStreamEvent(event="retrieved"),
                QueryStreamEvent(event="generating"),
                QueryStreamEvent(event="answer", data=self._answer.model_dump(mode="json")),
            ]
        for event in events:
            yield event


def _settings() -> Settings:
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
    )


def _client(service: FakeQueryService, *, db: FakeSession | None = None) -> TestClient:
    app = create_app(_settings())
    app.dependency_overrides[get_query_service] = lambda: service
    app.dependency_overrides[get_db_session] = lambda: db or FakeSession()
    return TestClient(app)


def _grounded() -> GroundedAnswer:
    return GroundedAnswer(
        answer="The magnetic chip detector traps ferrous particles.",
        citations=(
            Citation(
                source_id="faa-h-8083-32b-chapter-6",
                pdf_page_number=28,
                printed_page_label="6-28",
            ),
        ),
    )


def test_query_returns_the_grounded_answer_from_the_service() -> None:
    service = FakeQueryService(_grounded())

    response = _client(service).post("/v1/query", json={"question": "  chip detector?  "})

    assert response.status_code == 200
    assert response.json() == _grounded().model_dump(mode="json")
    assert service.questions == ["chip detector?"]


def test_query_passes_an_abstention_through_unchanged() -> None:
    service = FakeQueryService(abstention("The retrieved evidence does not answer this."))

    response = _client(service).post("/v1/query", json={"question": "capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["citations"] == []
    assert body["engine_health"] is None


def test_query_rejects_a_blank_question() -> None:
    service = FakeQueryService(_grounded())

    response = _client(service).post("/v1/query", json={"question": "   "})

    assert response.status_code == 422
    assert service.questions == []


def test_query_rejects_unknown_fields() -> None:
    response = _client(FakeQueryService(_grounded())).post(
        "/v1/query", json={"question": "ok", "evidence_limit": 9}
    )

    assert response.status_code == 422


def test_query_requires_a_request_body() -> None:
    response = _client(FakeQueryService(_grounded())).post("/v1/query")

    assert response.status_code == 422


def _read_sse(text: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE response body into ``(event, data)`` pairs."""
    events: list[tuple[str, dict[str, object]]] = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def test_query_stream_emits_stage_markers_then_the_answer() -> None:
    service = FakeQueryService(_grounded())

    response = _client(service).post("/v1/query/stream", json={"question": "chip detector?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _read_sse(response.text)
    assert [name for name, _ in events] == ["routed", "retrieved", "generating", "answer"]
    assert GroundedAnswer.model_validate(events[-1][1]) == _grounded()
    assert service.questions == ["chip detector?"]


def test_query_stream_reports_an_error_event() -> None:
    service = FakeQueryService(
        _grounded(),
        events=[
            QueryStreamEvent(event="error", data={"detail": "The query could not be completed."})
        ],
    )

    response = _client(service).post("/v1/query/stream", json={"question": "anything"})

    assert response.status_code == 200
    assert _read_sse(response.text) == [("error", {"detail": "The query could not be completed."})]


def test_query_stream_rejects_a_blank_question() -> None:
    response = _client(FakeQueryService(_grounded())).post(
        "/v1/query/stream", json={"question": "   "}
    )

    assert response.status_code == 422


def _only_run(db: FakeSession) -> QueryRun:
    assert len(db.added) == 1
    run = db.added[0]
    assert isinstance(run, QueryRun)
    return run


def test_query_records_a_query_run() -> None:
    db = FakeSession()

    _client(FakeQueryService(_grounded()), db=db).post(
        "/v1/query", json={"question": "chip detector?"}
    )

    run = _only_run(db)
    assert run.question == "chip detector?"
    assert run.abstained is False
    assert run.citation_count == 1
    assert run.citations[0]["pdf_page_number"] == 28
    assert run.latency_ms is not None
    assert db.committed is True


def test_query_still_answers_when_recording_fails() -> None:
    db = FakeSession(fail_on_commit=True)

    response = _client(FakeQueryService(_grounded()), db=db).post(
        "/v1/query", json={"question": "chip detector?"}
    )

    assert response.status_code == 200
    assert db.rolled_back is True


def test_query_stream_records_a_run_after_the_stream() -> None:
    db = FakeSession()

    _client(FakeQueryService(_grounded()), db=db).post(
        "/v1/query/stream", json={"question": "chip detector?"}
    )

    run = _only_run(db)
    assert run.question == "chip detector?"
    assert run.citation_count == 1
    assert db.committed is True
