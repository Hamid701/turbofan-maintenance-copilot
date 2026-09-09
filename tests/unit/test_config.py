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


def test_model_cache_defaults_to_the_repository_and_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    dsn = SecretStr("postgresql+psycopg://user:password@localhost:5432/turbofan")

    # A source checkout finds the weights under the repository's data directory.
    assert Settings(database_url=dsn).model_cache_dir.parts[-3:] == (
        "data",
        "processed",
        "models",
    )

    # A container installs the package into site-packages, where that relative
    # guess is meaningless, so the image sets the path explicitly.
    monkeypatch.setenv("TURBOFAN_MODEL_CACHE_DIR", "/opt/models")
    assert Settings(database_url=dsn).model_cache_dir == Path("/opt/models")


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


def test_openai_settings_are_optional_with_a_default_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    without_key = Settings(
        database_url=SecretStr("postgresql+psycopg://u:p@localhost:5432/turbofan"),
    )
    assert without_key.openai_api_key is None
    assert without_key.openai_model == "gpt-4o-mini"


def test_openai_key_and_model_load_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "TURBOFAN_DATABASE_URL",
        "postgresql+psycopg://u:p@localhost:5432/turbofan",
    )
    monkeypatch.setenv("TURBOFAN_OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("TURBOFAN_OPENAI_MODEL", "gpt-4o")

    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test-value"
    assert settings.openai_model == "gpt-4o"
