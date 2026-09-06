"""The process-wide SQLAlchemy engine, built lazily from validated settings."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from turbofan_copilot.core.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the shared engine for the configured database.

    The database secret is revealed only here, at the connection boundary.
    Tests that change the environment must call ``get_engine.cache_clear()``.
    """
    return create_engine(get_settings().database_url.get_secret_value())
