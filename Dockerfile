# syntax=docker/dockerfile:1.7
#
# Production image for the Turbofan Maintenance Copilot API.
#
# Two stages. The builder resolves the locked dependencies and downloads the
# pinned embedding weights; the runtime carries only the finished virtual
# environment, the weights, and the migrations, and runs as an unprivileged
# user. Nothing from the build - no uv, no compiler cache, no git history -
# survives into the shipped image.

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- build stage
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies resolve from the lockfile alone, so this layer is rebuilt only
# when pyproject.toml or uv.lock change - not on every source edit.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
# --no-editable installs the package properly into site-packages instead of
# linking back to /build/src, so the runtime stage needs no source tree.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Bake the embedding model in, using the id and revision pinned in the code, so
# the image is self-contained: no download on first request, and no dependence
# on Hugging Face being reachable in production.
ENV TURBOFAN_MODEL_CACHE_DIR=/opt/models
RUN /opt/venv/bin/python -c "\
from pathlib import Path; \
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder; \
BgeEmbedder(Path('/opt/models')); \
print('embedding weights cached')"

# -------------------------------------------------------------- runtime stage
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# An unprivileged user with no login shell. A container that is compromised
# through the application should not also be root inside it.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models
# Migrations ship with the image so `alembic upgrade head` can be run against a
# deployment. They are deliberately NOT applied at startup: several replicas
# booting at once must not race to migrate the same database.
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app alembic.ini ./

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TURBOFAN_MODEL_CACHE_DIR=/opt/models \
    HF_HUB_OFFLINE=1

USER app
EXPOSE 8000

# Liveness only: this asks whether the process is serving, not whether its
# dependencies are healthy. A readiness check that proves the database is
# reachable is the next increment.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "turbofan_copilot.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
