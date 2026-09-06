"""Typed application configuration loaded from environment variables."""

from enum import StrEnum
from functools import lru_cache

from pydantic import PostgresDsn, SecretStr, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_POSTGRES_DSN_ADAPTER = TypeAdapter(PostgresDsn)


class RuntimeEnvironment(StrEnum):
    """Deployment modes whose differences should be explicit in application code."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Validated settings populated from ``TURBOFAN_`` environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TURBOFAN_",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "Turbofan Maintenance Intelligence Copilot"
    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    database_url: SecretStr

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        """Require a PostgreSQL DSN while retaining secret-safe display behavior."""
        _POSTGRES_DSN_ADAPTER.validate_python(value.get_secret_value())
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once so one process observes one configuration snapshot."""
    # BaseSettings supplies required fields dynamically from environment sources.
    return Settings()  # type: ignore[call-arg]
