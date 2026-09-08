"""Tests for token accounting and the rough dollar estimate."""

from turbofan_copilot.llm.usage import TokenUsage, estimate_usd


def test_plus_accumulates_calls_and_tokens() -> None:
    total = (
        TokenUsage()
        .plus(prompt_tokens=100, completion_tokens=20)
        .plus(prompt_tokens=50, completion_tokens=10)
    )

    assert total.calls == 2
    assert total.prompt_tokens == 150
    assert total.completion_tokens == 30


def test_estimate_usd_uses_the_price_table() -> None:
    usage = TokenUsage(calls=1, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    # gpt-4o-mini is 0.15 prompt + 0.60 completion per million.
    assert estimate_usd("gpt-4o-mini", usage) == 0.75


def test_estimate_usd_is_none_for_an_unknown_model() -> None:
    usage = TokenUsage(calls=1, prompt_tokens=1000, completion_tokens=1000)

    assert estimate_usd("some-future-model", usage) is None


def test_estimate_usd_is_zero_for_no_usage() -> None:
    assert estimate_usd("gpt-4o-mini", TokenUsage()) == 0.0
