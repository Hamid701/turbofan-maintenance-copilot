"""Dependency checks behind the readiness endpoint.

Liveness and readiness answer different questions and must not be conflated.
*Liveness* asks whether the process is running; a failure means restart me.
*Readiness* asks whether this instance can do its job right now; a failure means
send traffic elsewhere, but leave me alone. A readiness probe that restarted the
container every time PostgreSQL blinked would turn a brief database outage into a
restart loop across every replica.

Each check is a small function returning a :class:`DependencyCheck`, and every one
of them catches its own failure: a probe that raises tells an orchestrator nothing
except that the endpoint is broken.
"""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.orm import Session

from turbofan_copilot.core.config import Settings
from turbofan_copilot.db.models import Chunk
from turbofan_copilot.db.session import get_engine

DATABASE_CHECK = "database"
CORPUS_CHECK = "corpus"
EMBEDDING_MODEL_CHECK = "embedding_model"
LLM_CREDENTIALS_CHECK = "llm_credentials"


class DependencyCheck(BaseModel):
    """The result of one readiness check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    ok: bool
    detail: str


def _failure(name: str, error: Exception) -> DependencyCheck:
    """Describe a failed check without leaking a connection string or a key.

    Exception text from a driver can carry a DSN, so only the exception type is
    reported; the full traceback belongs in the logs, not in an unauthenticated
    HTTP response.
    """
    return DependencyCheck(name=name, ok=False, detail=f"{type(error).__name__}")


def check_database(settings: Settings) -> DependencyCheck:
    """Confirm the database answers a trivial query."""
    try:
        with Session(get_engine(settings)) as session:
            session.execute(text("SELECT 1"))
    except Exception as error:
        return _failure(DATABASE_CHECK, error)
    return DependencyCheck(name=DATABASE_CHECK, ok=True, detail="reachable")


def check_corpus(settings: Settings) -> DependencyCheck:
    """Confirm the retrieval corpus has been ingested.

    A reachable but empty database is the failure mode of a fresh deployment: the
    service starts, answers liveness, and then abstains on every question because
    nothing can be retrieved. Readiness should catch that before traffic does.
    """
    try:
        with Session(get_engine(settings)) as session:
            count = session.query(Chunk).count()
    except Exception as error:
        return _failure(CORPUS_CHECK, error)
    if count == 0:
        return DependencyCheck(name=CORPUS_CHECK, ok=False, detail="no chunks ingested")
    return DependencyCheck(name=CORPUS_CHECK, ok=True, detail=f"{count} chunks")


def check_embedding_model(settings: Settings) -> DependencyCheck:
    """Confirm the embedding weights are present on disk.

    The container bakes them in and runs with ``HF_HUB_OFFLINE=1``, so a missing
    cache is a permanent failure rather than a slow first request.
    """
    cache = settings.model_cache_dir
    if not cache.is_dir() or not any(cache.iterdir()):
        return DependencyCheck(
            name=EMBEDDING_MODEL_CHECK, ok=False, detail="model cache is missing or empty"
        )
    return DependencyCheck(name=EMBEDDING_MODEL_CHECK, ok=True, detail="present")


def check_llm_credentials(settings: Settings) -> DependencyCheck:
    """Confirm a generation key is configured.

    Without one the service still serves ``/health`` and feedback, but every
    ``POST /v1/query`` fails. An instance that cannot answer questions should not
    be given question traffic, so this gates readiness rather than being advisory.
    """
    if settings.openai_api_key is None:
        return DependencyCheck(
            name=LLM_CREDENTIALS_CHECK, ok=False, detail="TURBOFAN_OPENAI_API_KEY is not set"
        )
    return DependencyCheck(name=LLM_CREDENTIALS_CHECK, ok=True, detail="configured")


READINESS_CHECKS: tuple[Callable[[Settings], DependencyCheck], ...] = (
    check_database,
    check_corpus,
    check_embedding_model,
    check_llm_credentials,
)


def run_readiness_checks(
    settings: Settings,
    checks: tuple[Callable[[Settings], DependencyCheck], ...] = READINESS_CHECKS,
) -> tuple[DependencyCheck, ...]:
    """Run every check and return all results, failures included.

    Deliberately not short-circuiting: an operator debugging a bad deployment
    wants the whole picture in one response, not the first thing that broke.
    """
    return tuple(check(settings) for check in checks)
