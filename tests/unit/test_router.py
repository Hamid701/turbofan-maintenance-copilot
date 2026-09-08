"""Tests for the engine-reference router."""

import pytest

from turbofan_copilot.llm.router import EngineReference, detect_engine_reference


@pytest.mark.parametrize(
    ("question", "expected_unit"),
    [
        ("Engine 47 shows a worsening health trend.", 47),
        ("what should I inspect on unit #3?", 3),
        ("ENGINE 100 exhaust gas temperature", 100),
        ("readings for engine 1 during start", 1),
    ],
)
def test_detects_a_valid_engine_number(question: str, expected_unit: int) -> None:
    assert detect_engine_reference(question) == EngineReference(split="test", unit_id=expected_unit)


@pytest.mark.parametrize(
    "question",
    [
        "How does a magnetic chip detector work?",
        "what are common engine failure modes?",
        "engine 0 is invalid",
        "engine 101 does not exist in FD001",
        "the 2010 engine overhaul manual",
    ],
)
def test_returns_none_when_no_valid_engine_is_named(question: str) -> None:
    assert detect_engine_reference(question) is None


def test_split_is_configurable() -> None:
    reference = detect_engine_reference("engine 12 history", split="train")
    assert reference == EngineReference(split="train", unit_id=12)
