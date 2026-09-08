"""Contract tests for the authenticated ingestion boundary."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.api.security import API_KEY_HEADER
from turbofan_copilot.core.config import RuntimeEnvironment, Settings

_KEY = "ingest-secret-value"


def _settings(*, ingest_key: str | None) -> Settings:
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
        ingest_api_key=SecretStr(ingest_key) if ingest_key is not None else None,
    )


def _client(*, ingest_key: str | None) -> TestClient:
    return TestClient(create_app(_settings(ingest_key=ingest_key)))


def test_ingest_accepts_a_request_with_the_configured_key() -> None:
    response = _client(ingest_key=_KEY).post(
        "/v1/ingest", json={"dataset": "corpus"}, headers={API_KEY_HEADER: _KEY}
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["dataset"] == "corpus"
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_ingest_rejects_a_missing_key() -> None:
    response = _client(ingest_key=_KEY).post("/v1/ingest", json={"dataset": "corpus"})

    assert response.status_code == 401
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_ingest_rejects_a_wrong_key() -> None:
    response = _client(ingest_key=_KEY).post(
        "/v1/ingest", json={"dataset": "fd001"}, headers={API_KEY_HEADER: "not-the-key"}
    )

    assert response.status_code == 401


def test_ingest_is_unavailable_when_no_key_is_configured() -> None:
    response = _client(ingest_key=None).post(
        "/v1/ingest", json={"dataset": "corpus"}, headers={API_KEY_HEADER: _KEY}
    )

    assert response.status_code == 503


def test_ingest_validates_the_dataset_name() -> None:
    response = _client(ingest_key=_KEY).post(
        "/v1/ingest", json={"dataset": "nonsense"}, headers={API_KEY_HEADER: _KEY}
    )

    assert response.status_code == 422
