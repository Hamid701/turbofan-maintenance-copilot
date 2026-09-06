"""Contract tests for the API liveness endpoint."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.core.config import RuntimeEnvironment, Settings


def test_health_returns_typed_liveness_response_without_secrets() -> None:
    database_url = "postgresql+psycopg://user:secret-value@localhost:5432/turbofan"
    settings = Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr(database_url),
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Turbofan Maintenance Intelligence Copilot",
        "environment": "test",
    }
    assert "secret-value" not in response.text
