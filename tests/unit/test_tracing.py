"""Langfuse tracing: off unless configured, and correctly nested when on.

Spans are captured in memory through the client's ``span_exporter`` parameter, so
nothing is sent over the network. The tests that matter most check the parent and
child relationships, because that is what breaks silently if the tracing context
fails to reach LangChain's executor thread or Starlette's thread pool.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from langfuse import LangfuseOtelSpanAttributes as Attr
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import get_db_session, get_query_service
from turbofan_copilot.api.schemas import QueryStreamEvent
from turbofan_copilot.api.security import API_KEY_HEADER
from turbofan_copilot.core.config import RuntimeEnvironment, Settings
from turbofan_copilot.core.tracing import (
    configure_tracing,
    flush_tracing,
    trace_question,
    trace_stage,
)
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.llm.answer import GroundedAnswer, ModelReply
from turbofan_copilot.llm.openai_provider import OpenAiProvider
from turbofan_copilot.llm.pipeline import build_answer_chain
from turbofan_copilot.retrieval.semantic_retrieval import ScoredChunk

API_KEY = "test-api-key"
QUESTION = "How does a magnetic chip detector reveal wear?"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": RuntimeEnvironment.TEST,
        "database_url": SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
        "client_api_key": SecretStr(API_KEY),
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Turn tracing on with an in-memory exporter, and off again afterwards."""
    exporter = InMemorySpanExporter()
    # Langfuse keeps one client per public key, so each test gets its own key.
    configure_tracing(
        _settings(
            langfuse_public_key=f"pk-lf-test-{uuid.uuid4().hex}",
            langfuse_secret_key=SecretStr("sk-lf-test"),
        ),
        span_exporter=exporter,
    )
    try:
        yield exporter
    finally:
        configure_tracing(_settings())


def _finished(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    flush_tracing()
    return {span.name: span for span in exporter.get_finished_spans()}


def _parent_name(span: ReadableSpan, by_name: dict[str, ReadableSpan]) -> str | None:
    if span.parent is None:
        return None
    for candidate in by_name.values():
        assert candidate.context is not None
        if candidate.context.span_id == span.parent.span_id:
            return candidate.name
    return "<unknown>"


def _provider() -> OpenAiProvider:
    """The real provider, talking to a fake OpenAI client."""
    completion = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=80),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=ModelReply(answerable=True, answer="ok", cited_passage_numbers=[1]),
                    refusal=None,
                )
            )
        ],
    )
    provider = OpenAiProvider("sk-test", "gpt-4o-mini")
    provider._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **_: completion))
    )
    return provider


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


# ------------------------------------------------------------------- off


def test_tracing_is_off_without_keys_and_every_call_is_a_no_op() -> None:
    assert configure_tracing(_settings()) is False

    with trace_question("req-1", endpoint="query", question=QUESTION) as root:
        root.update(output={"answer": "x"})
        with trace_stage("retrieve") as stage:
            stage.update(output=[])
    flush_tracing()  # nothing configured, nothing to send


# -------------------------------------------------------------------- on


def _assert_question_tree(by_name: dict[str, ReadableSpan], request_id: str) -> None:
    assert {"question", "route", "retrieve", "openai"} <= set(by_name)
    # Pinning the trace ID gives the root a placeholder parent that is never
    # exported; what matters is that none of our own spans sits above it.
    assert _parent_name(by_name["question"], by_name) in (None, "<unknown>")
    for child in ("route", "retrieve", "openai"):
        assert _parent_name(by_name[child], by_name) == "question"
    root = by_name["question"]
    assert root.context is not None
    assert format(root.context.trace_id, "032x") == Langfuse.create_trace_id(seed=request_id)
    generation = by_name["openai"].attributes or {}
    assert generation[Attr.OBSERVATION_TYPE] == "generation"
    assert generation[Attr.OBSERVATION_MODEL] == "gpt-4o-mini"
    assert json.loads(str(generation[Attr.OBSERVATION_USAGE_DETAILS])) == {
        "input": 1200,
        "output": 80,
    }
    retrieved = json.loads(str((by_name["retrieve"].attributes or {})[Attr.OBSERVATION_OUTPUT]))
    assert retrieved[0]["page"] == "6-28"


def test_a_blocking_question_is_one_trace_with_nested_stages_and_a_generation(
    spans: InMemorySpanExporter,
) -> None:
    chain = build_answer_chain(_OneChunkRetriever(), _provider())

    with trace_question("req-block", endpoint="query", question=QUESTION):
        chain.invoke(QUESTION)

    _assert_question_tree(_finished(spans), "req-block")


def test_a_streamed_question_nests_across_the_executor_thread(
    spans: InMemorySpanExporter,
) -> None:
    chain = build_answer_chain(_OneChunkRetriever(), _provider())

    async def run() -> None:
        with trace_question("req-stream", endpoint="stream", question=QUESTION):
            async for _ in chain.astream_events(QUESTION, version="v2"):
                pass

    asyncio.run(run())

    _assert_question_tree(_finished(spans), "req-stream")


# ------------------------------------------------------------ the routes


class _FakeService:
    def __init__(self, answer: GroundedAnswer | Exception) -> None:
        self._answer = answer

    def answer(self, question: str) -> GroundedAnswer:
        with trace_stage("retrieve", kind="retriever"):
            pass
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer

    async def answer_events(self, question: str) -> AsyncIterator[QueryStreamEvent]:
        with trace_stage("retrieve", kind="retriever"):
            pass
        assert isinstance(self._answer, GroundedAnswer)
        yield QueryStreamEvent(event="answer", data=self._answer.model_dump(mode="json"))


class _NullSession:
    def add(self, obj: object) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _client(service: _FakeService, exporter: InMemorySpanExporter) -> TestClient:
    app = create_app(_settings())
    # create_app configures tracing from settings (off); switch it back on here.
    configure_tracing(
        _settings(
            langfuse_public_key=f"pk-lf-test-{uuid.uuid4().hex}",
            langfuse_secret_key=SecretStr("sk-lf-test"),
        ),
        span_exporter=exporter,
    )
    app.dependency_overrides[get_query_service] = lambda: service
    app.dependency_overrides[get_db_session] = _NullSession
    return TestClient(app, headers={API_KEY_HEADER: API_KEY}, raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
def test_each_route_traces_the_question_and_sends_it(
    path: str, spans: InMemorySpanExporter
) -> None:
    client = _client(_FakeService(GroundedAnswer(answer="It traps debris.", citations=())), spans)

    response = client.post(path, json={"question": QUESTION}, headers={"X-Request-ID": "req-9"})

    assert response.status_code == 200
    # The route flushed on its own: the spans are already exported.
    by_name = {span.name: span for span in spans.get_finished_spans()}
    assert _parent_name(by_name["retrieve"], by_name) == "question"
    root = by_name["question"]
    assert root.context is not None
    assert format(root.context.trace_id, "032x") == Langfuse.create_trace_id(seed="req-9")
    output = json.loads(str((root.attributes or {})[Attr.OBSERVATION_OUTPUT]))
    assert output["answer"] == "It traps debris."


def test_a_failed_question_is_traced_as_an_error(spans: InMemorySpanExporter) -> None:
    client = _client(_FakeService(RuntimeError("model unavailable")), spans)

    response = client.post("/v1/query", json={"question": QUESTION})

    assert response.status_code == 500
    root = {span.name: span for span in spans.get_finished_spans()}["question"]
    attributes = root.attributes or {}
    assert attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
    assert attributes[Attr.OBSERVATION_STATUS_MESSAGE] == "RuntimeError"
