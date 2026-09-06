"""Shared fixtures for live-database integration tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from turbofan_copilot.db.session import get_engine
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder

MODEL_CACHE = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"


@pytest.fixture(scope="session")
def embedder() -> BgeEmbedder:
    """Load the pinned BGE model once for the whole integration session."""
    return BgeEmbedder(MODEL_CACHE)


@pytest.fixture(scope="session")
def db_engine() -> Engine:
    """Return the configured engine, or skip the test if PostgreSQL is unreachable."""
    engine = get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("PostgreSQL not reachable; run: docker compose up -d postgres")
    return engine


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Yield a session wrapped in a transaction that is always rolled back."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
