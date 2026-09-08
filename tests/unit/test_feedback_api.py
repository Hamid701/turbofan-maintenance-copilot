"""Contract tests for POST /v1/feedback."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.dependencies import get_db_session
from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.core.config import RuntimeEnvironment, Settings
from turbofan_copilot.db.models import Feedback


class FakeSession:
    """Records adds and commits without a database."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:  # pragma: no cover - feedback never rolls back
        pass


def _client(db: FakeSession | None = None) -> TestClient:
    settings = Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
    )
    app = create_app(settings)
    app.dependency_overrides[get_db_session] = lambda: db or FakeSession()
    return TestClient(app)


def test_feedback_is_accepted_and_recorded() -> None:
    db = FakeSession()

    response = _client(db).post(
        "/v1/feedback",
        json={"query_request_id": "abc123", "rating": "up", "comment": "  spot on  "},
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "accepted",
        "query_request_id": "abc123",
        "rating": "up",
        "request_id": response.headers[REQUEST_ID_HEADER],
    }
    assert len(db.added) == 1
    entry = db.added[0]
    assert isinstance(entry, Feedback)
    assert entry.rating == "up"
    assert entry.comment == "spot on"
    assert db.committed is True


def test_feedback_comment_is_optional() -> None:
    response = _client().post("/v1/feedback", json={"query_request_id": "abc123", "rating": "down"})

    assert response.status_code == 202


def test_feedback_rejects_an_unknown_rating() -> None:
    response = _client().post("/v1/feedback", json={"query_request_id": "abc123", "rating": "meh"})

    assert response.status_code == 422


def test_feedback_rejects_an_overlong_comment() -> None:
    response = _client().post(
        "/v1/feedback",
        json={"query_request_id": "abc123", "rating": "up", "comment": "x" * 2001},
    )

    assert response.status_code == 422


def test_feedback_rejects_unknown_fields() -> None:
    response = _client().post(
        "/v1/feedback",
        json={"query_request_id": "abc123", "rating": "up", "score": 5},
    )

    assert response.status_code == 422
