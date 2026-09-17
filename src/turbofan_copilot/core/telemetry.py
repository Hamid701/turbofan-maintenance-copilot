"""Per-question telemetry gathered across the API, pipeline, and model layers.

One question passes through three layers that do not know about each other: the
route, the LangChain pipeline, and the OpenAI provider. The pipeline knows how long
each stage took and the provider knows how many tokens were spent, but only the
route knows the request ID and writes the log line. Rather than thread a telemetry
object through every signature, the route opens a :class:`QueryTelemetry` in a
context variable and the lower layers add to it when one is present. This is the
mechanism OpenTelemetry uses for the current span.

A context variable follows the request into the threads and executor tasks that
serve it (Starlette's thread pool and LangChain's executor both copy the context),
and every request works in its own copy, so concurrent questions never mix their
numbers. Outside a request, for example in the evaluation script, nothing is open
and recording is a no-op.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class QueryTelemetry:
    """What one question cost: stage timings, model calls, tokens, and any error."""

    stage_ms: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_type: str | None = None

    def add_llm_call(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        """Fold one model call's token usage into this question's totals."""
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


_CURRENT: ContextVar[QueryTelemetry | None] = ContextVar("query_telemetry", default=None)


@contextmanager
def collect_query_telemetry() -> Iterator[QueryTelemetry]:
    """Open a collector for the question being served and close it afterwards."""
    telemetry = QueryTelemetry()
    token = _CURRENT.set(telemetry)
    try:
        yield telemetry
    finally:
        _CURRENT.reset(token)


def current_query_telemetry() -> QueryTelemetry | None:
    """Return the open collector, or ``None`` outside a served question."""
    return _CURRENT.get()


def elapsed_ms(started: float) -> int:
    """Whole milliseconds since ``started``, a ``perf_counter()`` reading."""
    return round((perf_counter() - started) * 1000)


@contextmanager
def timed_stage(name: str) -> Iterator[None]:
    """Record how long the enclosed block took as stage ``name``, if collecting."""
    started = perf_counter()
    try:
        yield
    finally:
        telemetry = _CURRENT.get()
        if telemetry is not None:
            telemetry.stage_ms[name] = elapsed_ms(started)
