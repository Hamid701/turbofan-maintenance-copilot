"""The generated OpenAPI schema carries request and response examples for /docs."""

from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.core.config import RuntimeEnvironment, Settings


def _openapi() -> Any:
    settings = Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def _request_schema(spec: Any, path: str) -> Any:
    body = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    ref = body["$ref"].removeprefix("#/components/schemas/")
    return spec["components"]["schemas"][ref]


def test_query_request_and_response_have_examples() -> None:
    spec = _openapi()

    assert _request_schema(spec, "/v1/query")["examples"][0]["question"]
    response = spec["paths"]["/v1/query"]["post"]["responses"]["200"]["content"]["application/json"]
    assert response["example"]["citations"][0]["printed_page_label"] == "6-28"


def test_ingest_and_feedback_requests_have_examples() -> None:
    spec = _openapi()

    assert _request_schema(spec, "/v1/ingest")["examples"][0] == {"dataset": "corpus"}
    assert _request_schema(spec, "/v1/feedback")["examples"][0]["rating"] == "up"
