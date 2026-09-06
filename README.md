# Turbofan Maintenance Intelligence Copilot

A learning-first Applied AI system for evidence-grounded turbofan maintenance support. The
project is currently in **Milestone 0: Foundation**.

This is an educational portfolio project, not certified aviation-maintenance software.

## Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop with the WSL 2 backend on Windows, or Docker Engine with Compose elsewhere

## Local setup

1. Create your untracked local configuration:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Replace `replace-with-local-password` in `.env`. Keep the password in
   `POSTGRES_PASSWORD` and `TURBOFAN_DATABASE_URL` synchronized.

3. Create or synchronize the Python environment from the lockfile:

   ```powershell
   uv sync --locked --group dev
   ```

## Start the foundation stack

Start PostgreSQL/pgvector and wait until its health check passes:

```powershell
docker compose up -d --wait postgres
```

Apply all database migrations:

```powershell
uv run --no-sync alembic upgrade head
```

Run the API during development:

```powershell
uv run --no-sync uvicorn turbofan_copilot.api.app:create_app --factory --app-dir src --reload
```

Verify the liveness endpoint from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Interactive OpenAPI documentation is available at <http://127.0.0.1:8000/docs>.

## Quality checks

```powershell
uv run --no-sync ruff check src tests migrations
uv run --no-sync ruff format --check src tests migrations
uv run --no-sync mypy src tests migrations
uv run --no-sync pytest
uv run --no-sync python -m build
```

Render migration SQL without changing a database:

```powershell
uv run --no-sync alembic upgrade head --sql
```

## Stop local services

Stop containers while preserving the PostgreSQL data volume:

```powershell
docker compose down
```

To intentionally delete the local database and start clean, use
`docker compose down --volumes`. This cannot recover data from the deleted volume.

## Current architecture boundary

- `pydantic-settings` validates `TURBOFAN_` environment variables.
- Docker Compose owns the PostgreSQL/pgvector server lifecycle.
- Alembic owns schema changes, beginning with activation of the `vector` extension.
- FastAPI exposes a typed `/health` liveness endpoint through an application factory.
- Retrieval, ingestion, engine-health modeling, and LLM orchestration are later milestones.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for project scope, decisions, evaluation plans,
safety boundaries, and the milestone roadmap.

## Windows and OneDrive note

OneDrive may mark virtual-environment package metadata as read-only, causing uv synchronization
errors. If that occurs, keep the repository outside a synchronized folder, delete only the
disposable `.venv`, and recreate it with `uv sync --locked --group dev`.
