"""Optional LLM tracing with Langfuse, behind a small interface that is a no-op when off.

A trace shows one question's path in full: the question, what routing decided, the
passages retrieved, the exact prompt and reply of the model call with its token
usage, and the final answer. Metrics say that something changed; a trace shows why.

The pipeline is instrumented at the boundaries it already has rather than through
LangChain's callback system: the model call is our own provider, not a LangChain
model, so a callback handler would not see it as a generation, and explicit spans
let us choose exactly what is sent to a third party.

Tracing is on only when :func:`configure_tracing` receives both Langfuse keys.
Otherwise every function here does nothing and ``langfuse`` is never imported, so
tests and local runs are unaffected. Langfuse is built on OpenTelemetry, whose
current span lives in a context variable: a span opened by the route becomes the
parent of spans opened in the pipeline and the provider, including across the
executor thread LangChain uses when streaming.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from turbofan_copilot.core.config import Settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export import SpanExporter

StageKind = Literal["span", "retriever", "chain", "tool"]

_client: "Langfuse | None" = None


class Observation:
    """What a traced block can add to its span; does nothing when tracing is off."""

    def __init__(self, handle: Any = None) -> None:
        self._handle = handle

    def update(self, **fields: Any) -> None:
        """Set fields such as ``output``, ``usage_details``, or ``level`` on the span."""
        if self._handle is not None:
            self._handle.update(**fields)


_OFF = Observation()


def configure_tracing(settings: Settings, *, span_exporter: "SpanExporter | None" = None) -> bool:
    """Turn tracing on if both Langfuse keys are configured; return whether it is on.

    ``span_exporter`` replaces the network exporter, which lets tests capture spans
    in memory.
    """
    global _client
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        _client = None
        return False
    from langfuse import Langfuse

    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.environment.value,
        span_exporter=span_exporter,
    )
    return True


@contextmanager
def trace_question(request_id: str, *, endpoint: str, question: str) -> Iterator[Observation]:
    """Open the root span for one question, with a trace ID derived from the request ID.

    The same request ID is in the ``query`` log event and the ``query_runs`` row, so
    one identifier finds a question in all three places.
    """
    if _client is None:
        yield _OFF
        return
    from langfuse import Langfuse, propagate_attributes

    trace_id = Langfuse.create_trace_id(seed=request_id)
    with (
        _client.start_as_current_observation(
            name="question",
            as_type="chain",
            input={"question": question},
            trace_context={"trace_id": trace_id},
        ) as root,
        propagate_attributes(
            trace_name="question", tags=[endpoint], metadata={"request_id": request_id}
        ),
    ):
        yield Observation(root)


@contextmanager
def trace_stage(name: str, *, kind: StageKind = "span", input: Any = None) -> Iterator[Observation]:
    """Open a child span for one pipeline stage."""
    if _client is None:
        yield _OFF
        return
    with _client.start_as_current_observation(name=name, as_type=kind, input=input) as span:
        yield Observation(span)


@contextmanager
def trace_generation(name: str, *, model: str, input: Any) -> Iterator[Observation]:
    """Open a generation span for one model call; record its output and token usage."""
    if _client is None:
        yield _OFF
        return
    with _client.start_as_current_observation(
        name=name, as_type="generation", model=model, input=input
    ) as generation:
        yield Observation(generation)


def flush_tracing() -> None:
    """Send queued spans now.

    Called after each question: with request-based billing Cloud Run gives an idle
    instance almost no CPU, so spans left to the background exporter could wait for
    the next request or be lost when the instance stops.
    """
    if _client is not None:
        _client.flush()
