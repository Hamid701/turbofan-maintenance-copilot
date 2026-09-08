"""Decide whether a question refers to one specific FD001 engine."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

Split = Literal["train", "test"]

_ENGINE_PATTERN = re.compile(r"\b(?:engine|unit)\s+#?(\d{1,3})\b", re.IGNORECASE)
_MIN_UNIT_ID = 1
_MAX_UNIT_ID = 100


class EngineReference(BaseModel):
    """A pointer to one engine's stored sensor history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Split
    unit_id: int


def detect_engine_reference(question: str, *, split: Split = "test") -> EngineReference | None:
    """Return the engine a question names, or None if it names no valid engine.

    Matches ``engine 47`` / ``unit #47`` (case-insensitive). FD001 has 100 engines
    per split, so an out-of-range number is treated as no reference.
    """
    match = _ENGINE_PATTERN.search(question)
    if match is None:
        return None
    unit_id = int(match.group(1))
    if not _MIN_UNIT_ID <= unit_id <= _MAX_UNIT_ID:
        return None
    return EngineReference(split=split, unit_id=unit_id)
