"""Database engines, cached per configured URL.

An engine belongs to a :class:`Settings` snapshot, not to the process: passing
settings explicitly is what lets an application (or a test) built around one
configuration be certain it is not talking to another one's database.
"""

import threading

from sqlalchemy import Engine, create_engine

from turbofan_copilot.core.config import Settings, get_settings

_ENGINES: dict[str, Engine] = {}
_LOCK = threading.Lock()


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the shared engine for ``settings``, or for the process-wide settings.

    The database secret is revealed only here, at the connection boundary. Engines
    are cached by URL so repeated calls reuse one connection pool.
    """
    resolved = settings or get_settings()
    url = resolved.database_url.get_secret_value()

    engine = _ENGINES.get(url)
    if engine is not None:
        return engine

    with _LOCK:
        # Another thread may have built it while we waited for the lock.
        existing = _ENGINES.get(url)
        if existing is not None:
            return existing
        created = create_engine(url)
        _ENGINES[url] = created
        return created


def dispose_engines() -> None:
    """Close every cached engine's pool and forget it (shutdown, and tests)."""
    with _LOCK:
        for engine in _ENGINES.values():
            engine.dispose()
        _ENGINES.clear()
