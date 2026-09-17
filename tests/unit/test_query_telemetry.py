"""Per-question telemetry: collected across layers, logged once, question left out.

The collector lives in a context variable, so the tests that matter most prove it
reaches the code that runs a question: LangChain's ``invoke`` and ``astream_events``
(which run steps in an executor) and Starlette's thread pool for the blocking route.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import (
    PipelineQueryService,
    get_db_session,
    get_query_service,
)
from turbofan_copilot.api.schemas import QueryStreamEvent
from turbofan_copilot.api.security import API_KEY_HEADER
from turbofan_copilot.core.config import RuntimeEnvironment, Settings
from turbofan_copilot.core.telemetry import (
    collect_query_telemetry,
    current_query_telemetry,
    timed_stage,
)
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.llm.answer import (
    NO_EVIDENCE_REASON,
    Citation,
    GroundedAnswer,
    ModelReply,
    abstention,
)
from turbofan_copilot.llm.openai_provider import OpenAiProvider
from turbofan_copilot.llm.pipeline import build_answer_chain
from turbofan_copilot.llm.provider import ChatMessage, ResponseModel
from turbofan_copilot.retrieval.semantic_retrieval import ScoredChunk

API_KEY = "test-api-key"
QUESTION = "How does a magnetic chip detector reveal wear?"


def _record_tokens(prompt: int = 1000, completion: int = 100) -> None:
    """Do what the OpenAI provider does after a call."""
    telemetry = current_query_telemetry()
    if telemetry is not None:
        telemetry.add_llm_call(prompt_tokens=prompt, completion_tokens=completion)


# ------------------------------------------------------------------ the collector


def test_nothing_is_recorded_outside_a_collector() -> None:
    with timed_stage("retrieve"):
        _record_tokens()

    assert current_query_telemetry() is None


def test_a_collector_gathers_stages_and_tokens_and_closes() -> None:
    with collect_query_telemetry() as telemetry:
        with timed_stage("retrieve"):
            pass
        _record_tokens(1000, 100)
        _record_tokens(500, 50)

    assert set(telemetry.stage_ms) == {"retrieve"}
    assert (telemetry.llm_calls, telemetry.prompt_tokens, telemetry.completion_tokens) == (
        2,
        1500,
        150,
    )
    assert current_query_telemetry() is None


# -------------------------------------------------------- reaching the pipeline


class _TokenRecordingProvider:
    def complete(
        self, messages: list[ChatMessage], *, response_model: type[ResponseModel]
    ) -> ResponseModel:
        _record_tokens()
        return response_model.model_validate(
            {"answerable": True, "answer": "ok", "cited_passage_numbers": [1]}
        )


class _OneChunkRetriever:
    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        chunk = DocumentChunk(
            source_id="faa-h-8083-32b-chapter-6",
            chapter_number=6,
            pdf_page_number=28,
            printed_page_label="6-28",
            chunk_index=1,
            text="chip detectors capture ferrous debris",
        )
        return ((0.9, chunk),)


def test_the_blocking_chain_records_every_stage_and_the_tokens() -> None:
    chain = build_answer_chain(_OneChunkRetriever(), _TokenRecordingProvider())

    with collect_query_telemetry() as telemetry:
        chain.invoke(QUESTION)

    assert set(telemetry.stage_ms) == {"route", "retrieve", "generate"}
    assert (telemetry.llm_calls, telemetry.prompt_tokens) == (1, 1000)


def test_the_streaming_chain_records_every_stage_and_the_tokens() -> None:
    # astream_events runs the steps in an executor, so this proves the context
    # variable is copied into it rather than lost.
    chain = build_answer_chain(_OneChunkRetriever(), _TokenRecordingProvider())

    async def run() -> None:
        async for _ in chain.astream_events(QUESTION, version="v2"):
            pass

    with collect_query_telemetry() as telemetry:
        asyncio.run(run())

    assert set(telemetry.stage_ms) == {"route", "retrieve", "generate"}
    assert (telemetry.llm_calls, telemetry.prompt_tokens) == (1, 1000)


def test_the_openai_provider_counts_tokens_per_question_and_in_total() -> None:
    completion = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=80),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=ModelReply(answerable=True, answer="ok", cited_passage_numbers=[]),
                    refusal=None,
                )
            )
        ],
    )
    provider = OpenAiProvider("sk-test", "gpt-4o-mini")
    provider._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **_: completion))
    )
    messages = [ChatMessage(role="user", content="q")]

    with collect_query_telemetry() as telemetry:
        provider.complete(messages, response_model=ModelReply)
    provider.complete(messages, response_model=ModelReply)  # outside: total only

    assert (telemetry.llm_calls, telemetry.prompt_tokens, telemetry.completion_tokens) == (
        1,
        1200,
        80,
    )
    assert provider.usage.calls == 2


def test_a_streaming_failure_records_its_type_without_logging_the_question(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(_: str) -> GroundedAnswer:
        raise TimeoutError("upstream timed out")

    service = PipelineQueryService.__new__(PipelineQueryService)
    service._chain = RunnableLambda(explode)

    async def run() -> list[QueryStreamEvent]:
        return [event async for event in service.answer_events(QUESTION)]

    create_app(_settings())  # installs the JSON log handler
    capsys.readouterr()
    with collect_query_telemetry() as telemetry:
        events = asyncio.run(run())

    assert [event.event for event in events] == ["error"]
    assert telemetry.error_type == "TimeoutError"
    assert QUESTION not in capsys.readouterr().out


# ------------------------------------------------------------- the query event


class _FakeService:
    def __init__(self, answer: GroundedAnswer | Exception) -> None:
        self._answer = answer

    def answer(self, question: str) -> GroundedAnswer:
        with timed_stage("generate"):
            _record_tokens()
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer

    async def answer_events(self, question: str) -> AsyncIterator[QueryStreamEvent]:
        with timed_stage("generate"):
            _record_tokens()
        assert isinstance(self._answer, GroundedAnswer)
        yield QueryStreamEvent(event="answer", data=self._answer.model_dump(mode="json"))


class _NullSession:
    def add(self, obj: object) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _settings() -> Settings:
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
        client_api_key=SecretStr(API_KEY),
        openai_model="gpt-4o-mini",
    )


def _client(service: _FakeService) -> TestClient:
    app = create_app(_settings())
    app.dependency_overrides[get_query_service] = lambda: service
    app.dependency_overrides[get_db_session] = _NullSession
    return TestClient(app, headers={API_KEY_HEADER: API_KEY}, raise_server_exceptions=False)


def _query_events(output: str) -> list[dict[str, object]]:
    lines = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    return [line for line in lines if line.get("event") == "query"]


def _cited() -> GroundedAnswer:
    return GroundedAnswer(
        answer="It captures ferrous debris.",
        citations=(
            Citation(
                source_id="faa-h-8083-32b-chapter-6", pdf_page_number=28, printed_page_label="6-28"
            ),
        ),
    )


@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
def test_each_question_writes_one_query_event_without_its_text(
    path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _client(_FakeService(_cited()))
    capsys.readouterr()

    response = client.post(path, json={"question": QUESTION}, headers={"X-Request-ID": "req-1"})

    output = capsys.readouterr().out
    events = _query_events(output)
    assert response.status_code == 200
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == "req-1"
    assert event["endpoint"] == ("stream" if path.endswith("stream") else "query")
    assert event["outcome"] == "answered"
    assert event["citation_count"] == 1
    assert event["llm_calls"] == 1
    assert event["prompt_tokens"] == 1000
    assert event["completion_tokens"] == 100
    assert event["model"] == "gpt-4o-mini"
    # gpt-4o-mini: 1000 x 0.15 + 100 x 0.60 per million tokens
    assert event["cost_usd"] == pytest.approx(0.00021)
    assert isinstance(event["stage_ms"], dict) and "generate" in event["stage_ms"]
    assert isinstance(event["latency_ms"], int)
    assert event["error_type"] is None
    assert QUESTION not in output


def test_a_refusal_is_logged_with_a_short_reason_code(capsys: pytest.CaptureFixture[str]) -> None:
    client = _client(_FakeService(abstention(NO_EVIDENCE_REASON)))
    capsys.readouterr()

    client.post("/v1/query", json={"question": QUESTION})

    event = _query_events(capsys.readouterr().out)[0]
    assert event["outcome"] == "abstained"
    assert event["abstention_reason"] == "no_evidence"


def test_a_failed_question_is_logged_with_its_error_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _client(_FakeService(RuntimeError("model unavailable")))
    capsys.readouterr()

    response = client.post("/v1/query", json={"question": QUESTION})

    event = _query_events(capsys.readouterr().out)[0]
    assert response.status_code == 500
    assert event["outcome"] == "error"
    assert event["error_type"] == "RuntimeError"
    assert event["llm_calls"] == 1
