"""Tests for the per-configuration database engine cache."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from turbofan_copilot.core.config import RuntimeEnvironment, Settings, get_settings
from turbofan_copilot.db.session import dispose_engines, get_engine


def _settings(database: str) -> Settings:
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr(f"postgresql+psycopg://user:password@localhost:5432/{database}"),
    )


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
    dispose_engines()

    engine = get_engine()
    try:
        assert engine is get_engine()
        assert engine.url.database == "engine-probe-db"
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        dispose_engines()
        get_settings.cache_clear()


def test_explicit_settings_win_over_the_process_wide_configuration() -> None:
    dispose_engines()
    try:
        explicit = get_engine(_settings("explicit-db"))
        assert explicit.url.database == "explicit-db"
        assert get_engine(_settings("explicit-db")) is explicit

        other = get_engine(_settings("another-db"))
        assert other is not explicit
        assert other.url.database == "another-db"
    finally:
        dispose_engines()
