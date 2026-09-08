"""Token accounting for an :class:`OpenAiProvider` and a rough dollar estimate.

The provider adds up the ``usage`` block OpenAI returns on every call. The price
table is small, hard-coded, and approximate - the estimate is "about this much",
not a bill. An unlisted model yields ``None`` rather than a wrong number.
"""

from pydantic import BaseModel, ConfigDict


class TokenUsage(BaseModel):
    """Cumulative prompt/completion tokens and call count for one provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def plus(self, *, prompt_tokens: int, completion_tokens: int) -> "TokenUsage":
        """Return a new total with one more call folded in."""
        return TokenUsage(
            calls=self.calls + 1,
            prompt_tokens=self.prompt_tokens + prompt_tokens,
            completion_tokens=self.completion_tokens + completion_tokens,
        )


# USD per 1,000,000 tokens as (prompt, completion), from the published OpenAI API
# pricing on 2026-09-07. Re-check before trusting a cost estimate months later.
_PRICE_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_usd(model: str, usage: TokenUsage) -> float | None:
    """Return a rough dollar cost for ``usage`` at ``model``'s price, or ``None``."""
    price = _PRICE_PER_MILLION.get(model)
    if price is None:
        return None
    prompt_price, completion_price = price
    dollars = (
        usage.prompt_tokens * prompt_price + usage.completion_tokens * completion_price
    ) / 1_000_000
    return round(dollars, 6)
