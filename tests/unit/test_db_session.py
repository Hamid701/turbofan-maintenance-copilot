"""Tests for the lazily built database engine."""

from pathlib import Path

import pytest

from turbofan_copilot.core.config import get_settings
from turbofan_copilot.db.session import get_engine


def test_engine_is_cached_and_targets_the_configured_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "TURBOFAN_DATABASE_URL",
        "postgresql+psycopg://user:password@localhost:5432/engine-probe-db",
    )
    get_settings.cache_clear()
    get_engine.cache_clear()

    engine = get_engine()
    try:
        assert engine is get_engine()
        assert engine.url.database == "engine-probe-db"
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        engine.dispose()
        get_engine.cache_clear()
        get_settings.cache_clear()
