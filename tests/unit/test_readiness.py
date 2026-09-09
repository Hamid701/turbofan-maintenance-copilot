"""Tests for readiness: the individual checks, and the endpoint that reports them.

No database and no network. Each check is exercised through its real failure path
by pointing settings at something that does not exist, and the endpoint is driven
with substituted checks so the 200/503 contract is tested independently.
"""

import time
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.readiness import (
    DependencyCheck,
    check_database,
    check_embedding_model,
    check_llm_credentials,
    run_readiness_checks,
)
from turbofan_copilot.core.config import RuntimeEnvironment, Settings

UNREACHABLE = "postgresql+psycopg://user:password@127.0.0.1:1/turbofan"


def settings_for(
    *,
    database_url: str = UNREACHABLE,
    model_cache_dir: Path | None = None,
    openai_key: str | None = None,
) -> Settings:
    """Build settings without touching the developer's real .env."""
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr(database_url),
        database_connect_timeout_seconds=2,
        model_cache_dir=model_cache_dir or Path("/nonexistent-model-cache"),
        openai_api_key=SecretStr(openai_key) if openai_key else None,
    )


def test_database_check_reports_failure_without_leaking_the_connection_string() -> None:
    check = check_database(settings_for())

    assert check.ok is False
    # The driver's error text can carry the DSN, so only the type is reported.
    assert "password" not in check.detail
    assert "127.0.0.1" not in check.detail


def test_database_check_fails_fast_instead_of_hanging() -> None:
    """An unreachable database must answer, not block.

    Without a connect timeout this call hung for minutes against a black-holed
    address, which would make an orchestrator kill the instance on probe timeout
    rather than route traffic away from it.
    """
    started = time.perf_counter()
    check = check_database(settings_for())
    elapsed = time.perf_counter() - started

    assert check.ok is False
    assert elapsed < 15


def test_embedding_model_check_notices_a_missing_or_empty_cache(tmp_path: Path) -> None:
    assert check_embedding_model(settings_for()).ok is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert check_embedding_model(settings_for(model_cache_dir=empty)).ok is False

    (empty / "weights.bin").write_bytes(b"x")
    present = check_embedding_model(settings_for(model_cache_dir=empty))
    assert present.ok is True
    assert present.detail == "present"


def test_llm_credential_check_gates_on_a_configured_key() -> None:
    assert check_llm_credentials(settings_for()).ok is False
    assert check_llm_credentials(settings_for(openai_key="sk-test")).ok is True


def test_every_check_runs_even_after_one_fails() -> None:
    # An operator debugging a bad deployment needs the whole picture at once.
    results = run_readiness_checks(settings_for())

    assert [check.name for check in results] == [
        "database",
        "corpus",
        "embedding_model",
        "llm_credentials",
    ]
    assert all(check.ok is False for check in results)


def ready_check(name: str) -> DependencyCheck:
    return DependencyCheck(name=name, ok=True, detail="fine")


def failing_check(name: str) -> DependencyCheck:
    return DependencyCheck(name=name, ok=False, detail="broken")


@pytest.mark.parametrize(
    ("checks", "expected_status", "expected_body_status"),
    [
        ((lambda _: ready_check("database"),), 200, "ready"),
        ((lambda _: failing_check("database"),), 503, "not_ready"),
        (
            (lambda _: ready_check("database"), lambda _: failing_check("corpus")),
            503,
            "not_ready",
        ),
    ],
)
def test_readiness_endpoint_answers_503_when_any_dependency_fails(
    monkeypatch: pytest.MonkeyPatch,
    checks: tuple[Callable[[Settings], DependencyCheck], ...],
    expected_status: int,
    expected_body_status: str,
) -> None:
    monkeypatch.setattr(
        "turbofan_copilot.api.routes.health.run_readiness_checks",
        lambda settings: tuple(check(settings) for check in checks),
    )

    with TestClient(create_app(settings_for()), raise_server_exceptions=False) as client:
        response = client.get("/ready")

    assert response.status_code == expected_status
    assert response.json()["status"] == expected_body_status
    assert len(response.json()["checks"]) == len(checks)


def test_liveness_stays_independent_of_dependency_health() -> None:
    # Everything below is unreachable, yet /health must still answer 200: the
    # process is alive, and restarting it would not fix a database outage.
    with TestClient(create_app(settings_for())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
