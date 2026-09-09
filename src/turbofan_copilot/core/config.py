"""Typed application configuration loaded from environment variables."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, SecretStr, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_POSTGRES_DSN_ADAPTER = TypeAdapter(PostgresDsn)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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

    # Optional so retrieval and engine-health work without an LLM. The pipeline
    # raises a clear error if it needs the key and it is absent.
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"

    # The SDK default is 600 s, long enough to pin a worker, a database session,
    # and the caller's connection for ten minutes on one hung upstream call.
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)

    # Shared secret for the ingestion endpoints. Absent means those endpoints are
    # unavailable (503), never open.
    ingest_api_key: SecretStr | None = None

    # Without this a connection attempt to an unreachable database blocks until
    # the operating system gives up, which can be minutes. That turns a readiness
    # probe into a hang and an orchestrator kills the instance on probe timeout
    # instead of being told it is simply not ready. libpq treats values below 2
    # as 2 seconds.
    database_connect_timeout_seconds: int = Field(default=5, ge=2)

    # Where the BGE weights live. The default is the repository's data directory,
    # which is correct for a source checkout; a container installs the package
    # into site-packages, where that relative guess is meaningless, so the image
    # sets TURBOFAN_MODEL_CACHE_DIR explicitly.
    model_cache_dir: Path = _REPOSITORY_ROOT / "data" / "processed" / "models"

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
