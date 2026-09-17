"""Tests for the browser chat page."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.core.config import RuntimeEnvironment, Settings


def settings() -> Settings:
    return Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:password@localhost:5432/turbofan"),
    )


def test_chat_page_is_served_as_html_and_talks_to_the_streaming_endpoint() -> None:
    with TestClient(create_app(settings())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/v1/query/stream" in response.text


def test_chat_page_inserts_model_output_as_text_not_html() -> None:
    # Answers quote retrieved documents, which are untrusted. The page must never
    # hand them to innerHTML, where a crafted passage could run script.
    with TestClient(create_app(settings())) as client:
        page = client.get("/").text

    assert "innerHTML" not in page
    assert "textContent" in page


def test_chat_page_sends_the_api_key_and_keeps_it_for_the_tab_only() -> None:
    # The /v1 endpoints need X-API-Key. The page keeps the key in sessionStorage,
    # which is cleared when the tab closes, never in localStorage, which persists.
    with TestClient(create_app(settings())) as client:
        page = client.get("/").text

    assert '"X-API-Key"' in page
    assert 'type="password"' in page
    assert "sessionStorage" in page
    assert "localStorage" not in page
    assert "response.status === 401" in page


def test_chat_page_is_not_part_of_the_api_contract() -> None:
    with TestClient(create_app(settings())) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
