"""Public response schemas for the API."""

from typing import Literal

from pydantic import BaseModel

from turbofan_copilot.core.config import RuntimeEnvironment


class HealthResponse(BaseModel):
    """Liveness information that is safe to expose without authentication."""

    status: Literal["ok"] = "ok"
    service: str
    environment: RuntimeEnvironment
