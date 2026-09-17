"""Contract tests for the X-API-Key boundary on the /v1 endpoints.

No model, no database, no network: the query service and the session are fakes
that record whether a rejected request ever reached them.
"""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import get_db_session, get_query_service
from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.api.schemas import QueryStreamEvent
from turbofan_copilot.api.security import API_KEY_HEADER
from turbofan_copilot.core.config import RuntimeEnvironment, Settings
from turbofan_copilot.llm.answer import GroundedAnswer, abstention

KEY = "correct-horse-battery-staple"

GATED = [
    ("/v1/query", {"question": "chip detector?"}, 200),
    ("/v1/query/stream", {"question": "chip detector?"}, 200),
    ("/v1/feedback", {"query_request_id": "abc123", "rating": "up"}, 202),
]


class _Recorder:
    """Counts every time a route got far enough to use its dependencies."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str) -> GroundedAnswer:
        return abstention("not covered")

    async def answer_events(self, question: str) -> AsyncIterator[QueryStreamEvent]:
        yield QueryStreamEvent(event="answer", data=self.answer(question).model_dump(mode="json"))

    def add(self, obj: object) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def dependency(self) -> "_Recorder":
        self.calls += 1
        return self


def _client(recorder: _Recorder, *, key: str | None = KEY) -> TestClient:
    settings = Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
        client_api_key=SecretStr(key) if key is not None else None,
    )
    app = create_app(settings)
    app.dependency_overrides[get_query_service] = recorder.dependency
    app.dependency_overrides[get_db_session] = recorder.dependency
    return TestClient(app)


@pytest.mark.parametrize(("path", "body", "_status"), GATED)
def test_a_missing_key_is_rejected_before_the_route_runs(
    path: str, body: dict[str, str], _status: int
) -> None:
    recorder = _Recorder()

    response = _client(recorder).post(path, json=body)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "APIKey"
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]
    # Neither the pipeline nor the database was touched.
    assert recorder.calls == 0


@pytest.mark.parametrize(
    "supplied",
    [b"wrong-key", KEY.upper().encode(), KEY.encode() + b"x", b"", b"caf\xe9"],
    ids=["wrong", "case", "longer", "empty", "non-ascii"],
)
def test_a_wrong_key_is_rejected(supplied: bytes) -> None:
    recorder = _Recorder()

    response = _client(recorder).post(
        "/v1/query",
        json={"question": "chip detector?"},
        headers={API_KEY_HEADER.encode(): supplied},
    )

    # A non-ASCII byte must be a 401, not a TypeError from compare_digest.
    assert response.status_code == 401
    assert recorder.calls == 0


@pytest.mark.parametrize(("path", "body", "_status"), GATED)
def test_no_configured_key_closes_the_endpoints(
    path: str, body: dict[str, str], _status: int
) -> None:
    recorder = _Recorder()

    response = _client(recorder, key=None).post(path, json=body, headers={API_KEY_HEADER: KEY})

    assert response.status_code == 503
    assert recorder.calls == 0


@pytest.mark.parametrize(("path", "body", "status"), GATED)
def test_the_correct_key_is_let_through(path: str, body: dict[str, str], status: int) -> None:
    response = _client(_Recorder()).post(path, json=body, headers={API_KEY_HEADER: KEY})

    assert response.status_code == status


def test_authentication_is_checked_before_the_body_is_validated() -> None:
    # An anonymous caller learns nothing about the request schema.
    response = _client(_Recorder()).post("/v1/query", json={"question": "   "})

    assert response.status_code == 401


def test_health_and_the_chat_page_need_no_key() -> None:
    client = _client(_Recorder(), key=None)

    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_openapi_declares_the_key_on_gated_routes_only() -> None:
    spec = _client(_Recorder()).get("/openapi.json").json()

    scheme = spec["components"]["securitySchemes"]["APIKeyHeader"]
    assert scheme == {"type": "apiKey", "in": "header", "name": API_KEY_HEADER}
    for path, _, _ in GATED:
        assert spec["paths"][path]["post"]["security"] == [{"APIKeyHeader": []}]
    assert "security" not in spec["paths"]["/health"]["get"]
