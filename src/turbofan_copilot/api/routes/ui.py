"""The browser chat page.

A single self-contained HTML file with no build step and no external assets. It
talks to ``POST /v1/query/stream`` from the same origin, so it needs no CORS
configuration and deploys inside the API container at no extra cost.
"""

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# Read through importlib.resources rather than a path relative to this file: the
# production image installs the package into site-packages, where only files
# declared as package data exist.
_PAGE = files("turbofan_copilot.api").joinpath("static", "index.html")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def get_chat_page() -> HTMLResponse:
    """Serve the chat page. Read per request so edits show without a restart."""
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))
