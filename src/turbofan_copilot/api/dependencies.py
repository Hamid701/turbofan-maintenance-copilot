"""Shared, expensively-built objects the API routes depend on.

The grounded-answer pipeline needs an embedding model, the persisted corpus, a
fitted degradation model, and an LLM client. Building those per request would be
wasteful, so ``get_query_service`` builds one :class:`PipelineQueryService` the
first time a request needs it and caches it on ``app.state`` for the rest of the
process. Tests replace the whole service through ``app.dependency_overrides``.
"""

import logging
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Protocol, cast

from fastapi import FastAPI, Request
from sqlalchemy.orm import Session, sessionmaker

from turbofan_copilot.api.schemas import QueryStreamEvent, QueryStreamEventName
from turbofan_copilot.core.config import Settings
from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.llm.answer import GroundedAnswer
from turbofan_copilot.llm.openai_provider import build_openai_provider
from turbofan_copilot.llm.pipeline import build_answer_chain
from turbofan_copilot.llm.router import EngineReference
from turbofan_copilot.llm.tools import (
    EngineHealthReport,
    build_engine_health_report,
    load_degradation_model,
    load_rul_regressor,
)
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import HybridRetriever
from turbofan_copilot.retrieval.semantic_retrieval import SemanticRetriever

_MODEL_CACHE = Path(__file__).resolve().parents[3] / "data" / "processed" / "models"

_logger = logging.getLogger(__name__)

# The pipeline steps whose completion is worth streaming to the client. The LCEL
# chain names each RunnableLambda after its function, so these are stable.
_PROGRESS_EVENTS: dict[tuple[str, str], QueryStreamEventName] = {
    ("on_chain_end", "route"): "routed",
    ("on_chain_end", "retrieve"): "retrieved",
    ("on_chain_start", "generate"): "generating",
}


class QueryService(Protocol):
    """Answers one maintenance question with citations, or an abstention."""

    def answer(self, question: str) -> GroundedAnswer: ...

    def answer_events(self, question: str) -> AsyncIterator[QueryStreamEvent]: ...


class PipelineQueryService:
    """Wire the grounded-answer pipeline once, then answer many questions.

    Retrieval runs fully in memory: the corpus and its vectors are loaded from
    PostgreSQL at construction and never queried again. A short-lived session is
    opened only when a question names an engine and the FD001 health tools run.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise RuntimeError("TURBOFAN_OPENAI_API_KEY must be set to serve POST /v1/query")
        self._provider = build_openai_provider(settings)
        self._session_factory = sessionmaker(get_engine(settings))

        embedder = BgeEmbedder(_MODEL_CACHE)
        with self._session_factory() as session:
            corpus = load_persisted_corpus(session)
            self._degradation_model = load_degradation_model(session)
            # Fitted once at construction, like the retriever: a few seconds over
            # the stored train set, then every request reuses it.
            self._rul_regressor = load_rul_regressor(session)

        lexical = LexicalRetriever(corpus.chunks)
        semantic = SemanticRetriever(corpus.chunks, corpus.vectors, embedder)
        retriever = HybridRetriever(lexical, semantic, corpus_size=len(corpus.chunks))

        # One chain, built once, drives both the blocking and the streaming route.
        self._chain = build_answer_chain(
            retriever, self._provider, engine_report_lookup=self._engine_report
        )

    def _engine_report(self, reference: EngineReference) -> EngineHealthReport:
        with self._session_factory() as session:
            return build_engine_health_report(
                session,
                reference,
                degradation_model=self._degradation_model,
                rul_regressor=self._rul_regressor,
            )

    def answer(self, question: str) -> GroundedAnswer:
        """Retrieve manual evidence, attach engine data if named, and answer."""
        return self._chain.invoke(question)

    async def answer_events(self, question: str) -> AsyncIterator[QueryStreamEvent]:
        """Run the chain, streaming a marker per stage, then the final answer.

        The model call cannot stream partial JSON (structured output), so the
        markers report progress and the ``answer`` event carries the whole
        ``GroundedAnswer`` - identical to what ``POST /v1/query`` returns.
        """
        final: GroundedAnswer | None = None
        try:
            async for event in self._chain.astream_events(question, version="v2"):
                marker = _PROGRESS_EVENTS.get((event["event"], event["name"]))
                if marker is not None:
                    yield QueryStreamEvent(event=marker)
                output = event["data"].get("output")
                if isinstance(output, GroundedAnswer):
                    final = output
        except Exception:
            _logger.exception("streaming query failed for question %r", question)
            yield QueryStreamEvent(
                event="error", data={"detail": "The query could not be completed."}
            )
            return

        if final is None:  # pragma: no cover - the chain always yields an answer
            yield QueryStreamEvent(event="error", data={"detail": "No answer was produced."})
            return
        yield QueryStreamEvent(event="answer", data=final.model_dump(mode="json"))


_BUILD_LOCK = threading.Lock()


def build_query_service(app: FastAPI) -> QueryService:
    """Build the query service once for ``app`` and cache it on ``app.state``.

    Concurrent first requests must not each load the embedding model, and a build
    that fails (no key, no corpus) must not be retried on every request - the
    failure is cached and re-raised, so the cause is reported once and cheaply.
    """
    state = app.state
    with _BUILD_LOCK:
        failure = getattr(state, "query_service_error", None)
        if failure is not None:
            raise failure
        service = getattr(state, "query_service", None)
        if service is None:
            try:
                service = PipelineQueryService(cast(Settings, state.settings))
            except Exception as error:
                state.query_service_error = error
                _logger.exception("query service could not be built")
                raise
            state.query_service = service
    return cast(QueryService, service)


def get_query_service(request: Request) -> QueryService:
    """Return the process-wide query service, building it on first use."""
    return build_query_service(request.app)


def get_db_session(request: Request) -> Iterator[Session]:
    """Yield a short-lived session on *this app's* database; the route commits."""
    settings = cast(Settings, request.app.state.settings)
    with Session(get_engine(settings)) as session:
        yield session


__all__ = [
    "PipelineQueryService",
    "QueryService",
    "build_query_service",
    "get_db_session",
    "get_query_service",
]
