"""Structured logging: one JSON object per line, fields that can be queried."""

import json
import logging
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from turbofan_copilot.api.app import create_app
from turbofan_copilot.api.middleware import REQUEST_ID_HEADER
from turbofan_copilot.core.config import LogLevel, RuntimeEnvironment, Settings
from turbofan_copilot.core.logging_setup import PACKAGE_LOGGER, JsonFormatter, configure_logging


def _record(message: str, *, fields: object = None, level: int = logging.INFO) -> logging.LogRecord:
    record = logging.LogRecord("turbofan_copilot.test", level, __file__, 1, message, None, None)
    if fields is not None:
        record.fields = fields
    return record


def test_a_record_becomes_one_json_line_with_severity_message_and_fields() -> None:
    line = JsonFormatter().format(
        _record("GET /health -> 200", fields={"status": 200, "latency_ms": 3})
    )

    assert "\n" not in line
    assert json.loads(line) == {
        "severity": "INFO",
        "message": "GET /health -> 200",
        "logger": "turbofan_copilot.test",
        "status": 200,
        "latency_ms": 3,
    }


def test_a_newline_in_the_message_cannot_start_a_second_log_entry() -> None:
    # Text such as a question can contain a newline; in plain-text logs that would
    # forge a separate entry. JSON escapes it, so the output stays one line.
    line = JsonFormatter().format(_record('ok\n{"severity": "CRITICAL", "message": "forged"}'))

    assert "\n" not in line
    assert json.loads(line)["severity"] == "INFO"


def test_an_exception_is_included_on_the_same_line() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = _record("failed", level=logging.ERROR)
        record.exc_info = sys.exc_info()

    entry = json.loads(JsonFormatter().format(record))

    assert entry["severity"] == "ERROR"
    assert "RuntimeError: boom" in entry["exception"]


def test_configuring_twice_installs_one_handler_and_applies_the_level() -> None:
    logger = logging.getLogger(PACKAGE_LOGGER)
    configure_logging(LogLevel.WARNING)
    configure_logging(LogLevel.DEBUG)

    try:
        assert len(logger.handlers) == 1
        assert logger.level == logging.DEBUG
    finally:
        configure_logging(LogLevel.INFO)


def test_each_request_writes_one_structured_line(capsys: pytest.CaptureFixture[str]) -> None:
    settings = Settings(
        environment=RuntimeEnvironment.TEST,
        database_url=SecretStr("postgresql+psycopg://user:pw@localhost:5432/turbofan"),
    )
    client = TestClient(create_app(settings))
    capsys.readouterr()

    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-123"})

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    requests = [line for line in lines if line["logger"] == "turbofan_copilot.api.request"]
    assert response.status_code == 200
    assert len(requests) == 1
    entry = requests[0]
    assert entry["severity"] == "INFO"
    assert entry["message"] == "GET /health -> 200"
    assert entry["request_id"] == "trace-123"
    assert entry["method"] == "GET"
    assert entry["path"] == "/health"
    assert entry["status"] == 200
    assert isinstance(entry["latency_ms"], int)
