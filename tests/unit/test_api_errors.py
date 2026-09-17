"""Contract tests for request-ID correlation and structured error bodies."""

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import get_db_session, get_query_service
from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.api.security import API_KEY_HEADER
from turbofan_copilot.core.config import RuntimeEnvironment, Settings
from turbofan_copilot.llm.answer import GroundedAnswer

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
API_KEY = "test-api-key"


def _settings() -> Settings:
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
        client_api_key=SecretStr(API_KEY),
    )


class _BoomService:
    """A query service whose blocking path always raises."""

    def answer(self, question: str) -> GroundedAnswer:
        raise RuntimeError("pipeline exploded")

    def answer_events(self, question: str) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


class _NullSession:
    """A session stand-in so no unit test can reach a real database."""

    def add(self, obj: object) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _query_app(service: object) -> FastAPI:
    """An app whose query route is fully stubbed - no model, no database."""
    app = create_app(_settings())
    app.dependency_overrides[get_query_service] = lambda: service
    app.dependency_overrides[get_db_session] = _NullSession
    return app


def test_health_response_carries_a_generated_request_id() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert _HEX32.match(response.headers[REQUEST_ID_HEADER])


def test_a_supplied_request_id_is_echoed_back() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"


def test_an_unhandled_error_becomes_a_structured_500() -> None:
    client = TestClient(
        _query_app(_BoomService()),
        raise_server_exceptions=False,
        headers={API_KEY_HEADER: API_KEY},
    )

    response = client.post("/v1/query", json={"question": "how do compressors work?"})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error."
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert _HEX32.match(body["request_id"])


def test_a_validation_error_keeps_field_detail_and_adds_the_request_id() -> None:
    with TestClient(_query_app(_BoomService()), headers={API_KEY_HEADER: API_KEY}) as client:
        response = client.post("/v1/query", json={"question": "   "})

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], list)
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
