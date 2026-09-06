"""Tests for the typed application configuration boundary."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from turbofan_copilot.core.config import (
    LogLevel,
    RuntimeEnvironment,
    Settings,
    get_settings,
)


def test_settings_use_safe_application_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        database_url=SecretStr("postgresql+psycopg://user:password@localhost:5432/turbofan"),
    )

    assert settings.environment is RuntimeEnvironment.DEVELOPMENT
    assert settings.log_level is LogLevel.INFO
    assert str(settings.database_url) == "**********"


def test_settings_load_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TURBOFAN_ENVIRONMENT", "test")
    monkeypatch.setenv("TURBOFAN_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv(
        "TURBOFAN_DATABASE_URL",
        "postgresql+psycopg://test-user:test-password@localhost:5432/test-db",
    )

    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()

    assert settings.environment is RuntimeEnvironment.TEST
    assert settings.log_level is LogLevel.DEBUG
    assert settings.database_url.get_secret_value().endswith("/test-db")


def test_settings_reject_non_postgresql_database_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(
            database_url=SecretStr("sqlite:///local.db"),
        )
