"""An :class:`LlmProvider` backed by the OpenAI chat completions API."""

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from turbofan_copilot.core.config import Settings
from turbofan_copilot.llm.provider import ChatMessage, ResponseModel
from turbofan_copilot.llm.usage import TokenUsage


def _to_param(message: ChatMessage) -> ChatCompletionMessageParam:
    """Convert our narrow message type into the OpenAI SDK's typed dict."""
    if message.role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    return ChatCompletionUserMessageParam(role="user", content=message.content)


class OpenAiProvider:
    """Call one OpenAI chat model and parse the reply into a Pydantic model.

    Uses structured outputs (``response_format`` with a schema), so the reply
    always validates or the call raises.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model = model
        self._usage = TokenUsage()

    @property
    def model(self) -> str:
        """The chat model this provider calls."""
        return self._model

    @property
    def usage(self) -> TokenUsage:
        """Cumulative token usage across every ``complete`` call so far."""
        return self._usage

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """Send ``messages`` and return the parsed structured reply."""
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[_to_param(message) for message in messages],
            response_format=response_model,
        )
        if completion.usage is not None:
            self._usage = self._usage.plus(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            refusal = completion.choices[0].message.refusal
            raise RuntimeError(f"model returned no parseable content (refusal: {refusal})")
        return parsed


def build_openai_provider(settings: Settings) -> OpenAiProvider:
    """Build the provider from settings so the timeout is never left at the default."""
    if settings.openai_api_key is None:
        raise RuntimeError("TURBOFAN_OPENAI_API_KEY must be set to call the language model")
    return OpenAiProvider(
        settings.openai_api_key.get_secret_value(),
        settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
