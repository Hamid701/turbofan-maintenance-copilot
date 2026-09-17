"""Structured JSON logging for the application's own loggers.

Every logger in the package lives under ``turbofan_copilot``. Without a handler
Python prints only WARNING and above through its last-resort handler, so INFO lines
such as the per-request log were silently dropped. This module gives the package
logger one handler that writes a single JSON object per line to stdout.

Cloud Run turns a JSON line on stdout into a structured entry: ``severity`` sets the
entry's severity, ``message`` is the text shown in the log viewer, and every other
key becomes a queryable field of ``jsonPayload``. JSON encoding also keeps each
entry on one line, so text containing a newline cannot forge a second log entry.

Structured fields are passed as ``extra={"fields": {...}}``, one attribute that
cannot collide with the standard ``LogRecord`` attributes.
"""

import json
import logging
import sys

from turbofan_copilot.core.config import LogLevel

PACKAGE_LOGGER = "turbofan_copilot"


class JsonFormatter(logging.Formatter):
    """Render a record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class _JsonHandler(logging.Handler):
    """Write each record to the current ``sys.stdout``.

    The stream is looked up on every write rather than captured once, because
    test runners replace ``sys.stdout`` and a handler holding an old stream would
    write to a closed file. The class also marks the handler this module installs.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def configure_logging(level: LogLevel) -> None:
    """Send the package's logs to stdout as JSON at ``level``.

    Safe to call repeatedly (the app factory runs once per app, and tests build
    many apps): the handler is installed once and only the level is updated. The
    root logger is left alone, so a host's own handlers, such as pytest's log
    capture, keep receiving records through propagation.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level.value)
    if not any(isinstance(handler, _JsonHandler) for handler in logger.handlers):
        handler = _JsonHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
