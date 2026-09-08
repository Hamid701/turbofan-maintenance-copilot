"""The narrow boundary the answer pipeline uses to call a language model."""

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ChatMessage(BaseModel):
    """One message in a chat completion request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user"]
    content: str


class LlmProvider(Protocol):
    """Anything that turns chat messages into a validated instance of a Pydantic model.

    Keeping the interface this small means the pipeline never imports a provider
    SDK, and unit tests can supply a fixed reply.
    """

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """Return the model's reply parsed into ``response_model``."""
        ...
