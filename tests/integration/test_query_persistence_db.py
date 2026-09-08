"""The query and feedback endpoints write rows against a real PostgreSQL."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import get_db_session
from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.core.config import get_settings
from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.models import Feedback, QueryRun

pytestmark = pytest.mark.integration

EXPECTED_CHUNKS = 1050


def _app_with(session: Session) -> TestClient:
    app = create_app(get_settings())
    app.dependency_overrides[get_db_session] = lambda: session
    return TestClient(app)


def test_feedback_endpoint_writes_a_row(db_session: Session) -> None:
    client = _app_with(db_session)

    response = client.post(
        "/v1/feedback",
        json={"query_request_id": "req-xyz", "rating": "down", "comment": "missed the point"},
    )

    assert response.status_code == 202
    row = (
        db_session.execute(select(Feedback).where(Feedback.query_request_id == "req-xyz"))
        .scalars()
        .one()
    )
    assert row.rating == "down"
    assert row.comment == "missed the point"
    assert row.created_at is not None


def test_query_endpoint_writes_a_run(db_engine: Engine, db_session: Session) -> None:
    settings = get_settings()
    if settings.openai_api_key is None:
        pytest.skip("TURBOFAN_OPENAI_API_KEY not set")
    if len(load_persisted_corpus(db_session).chunks) != EXPECTED_CHUNKS:
        pytest.skip("700-char corpus not ingested")

    client = _app_with(db_session)
    response = client.post(
        "/v1/query",
        json={"question": "How does a magnetic chip detector reveal internal engine wear?"},
    )

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]
    run = (
        db_session.execute(select(QueryRun).where(QueryRun.request_id == request_id))
        .scalars()
        .one()
    )
    assert run.question.startswith("How does a magnetic chip detector")
    assert run.citation_count == len(response.json()["citations"])
    assert run.latency_ms is not None and run.latency_ms >= 0
