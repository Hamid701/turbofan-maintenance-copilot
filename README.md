# Turbofan Maintenance Intelligence Copilot

A retrieval-augmented maintenance assistant for turbine engines. It answers questions from
the FAA *Aviation Maintenance Technician Handbook — Powerplant* with **page-level citations**,
pulls a **deterministic engine-health report** from NASA C-MAPSS sensor data when a question
names an engine, and **refuses to answer** when the manual does not cover the question.

This is an educational portfolio project, not certified aviation-maintenance software. Nothing
here should be used to make a real airworthiness decision.

## What it does

Ask a question and get an answer you can check:

```bash
curl -X POST localhost:8000/v1/query -H "content-type: application/json" \
  -d '{"question":"How does a magnetic chip detector reveal internal engine wear?"}'
```

```json
{
  "answer": "A magnetic chip detector sits in the oil system and captures ferrous debris...",
  "citations": [
    {"source_id": "faa-h-8083-32b-chapter-6", "pdf_page_number": 28, "printed_page_label": "6-28"}
  ],
  "engine_health": null,
  "abstained": false,
  "abstention_reason": null
}
```

Three behaviours are the point of the project:

- **Citations are reconstructed, never generated.** The model returns passage *numbers*; the
  pipeline maps those back to the retrieved chunks. A page reference can therefore not be
  hallucinated.
- **It abstains.** Out-of-corpus questions and prompt-injection attempts return
  `abstained: true` with no answer text. Scored 10/10 on a labelled evaluation set.
- **Engine data is deterministic.** "Engine 5 shows rising EGT during start — what should I
  inspect?" attaches a least-squares sensor-trend summary and a remaining-useful-life estimate
  computed in plain Python, clearly separated from the manual's guidance.

## Measured results

Retrieval over nine frozen, page-grounded questions (top-5 budget):

| Config | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Lexical (IDF baseline) | 0.556 | **0.889** | **0.889** | 0.722 |
| BGE semantic | 0.667 | 0.889 | 0.889 | 0.759 |
| **RRF hybrid (shipped)** | **0.778** | 0.778 | 0.778 | **0.778** |

The hybrid wins on Hit@1 and MRR and is the shipped configuration. It *loses* Hit@3/5 to the
lexical baseline — two paraphrased questions fall to ranks 7 and 11. That is recorded rather
than hidden; see the decision log in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

- **Abstention:** 10/10 on 10 labelled cases (4 answerable, 3 out-of-corpus, 3 injection).
- **RUL baseline:** test-set MAE 25.8 / RMSE 32.7 cycles, against a naive mean predictor's
  35.9 / 40.1.
- **Cost:** a full evaluation run is roughly US$0.002 on `gpt-4o-mini`.

**Honest limits.** Nine retrieval cases and ten abstention cases are a smoke test, not a
benchmark — one case is worth 11% of a retrieval metric. The abstention cases are textbook
injections, not adversarial ones. The RUL model is a linear degradation-index extrapolation,
not a competitive C-MAPSS entry.

## Architecture

```
FAA PDFs ──▶ page-preserving extraction ──▶ 700-char chunks ──▶ BGE embeddings
                                                    │                  │
                                                    ▼                  ▼
                                            PostgreSQL 17  +  pgvector (HNSW)
                                                    │
              ┌─────────────────────────────────────┴───────────────┐
              ▼                                                     ▼
   lexical (IDF)  +  semantic (cosine)  ──▶ RRF hybrid       FD001 sensor readings
                                              │                     │
                                              ▼                     ▼
                          LangChain LCEL chain: route ▸ retrieve ▸ generate ▸ cite
                                              │      (two refusal branches)
                                              ▼
                                    FastAPI  ──▶  query_runs / feedback
```

| Layer | Choice | Why |
|---|---|---|
| Chunking | 700 chars / 100 overlap, page-contained | Page provenance is what makes a citation checkable |
| Embeddings | `BAAI/bge-small-en-v1.5`, pinned revision, local CPU | Reproducible, no embedding API cost |
| Vector store | PostgreSQL + pgvector, HNSW `vector_cosine_ops` | Relational and vector data in one place |
| Fusion | Reciprocal rank fusion, `k = 60` | No score normalisation needed across two scales |
| Orchestration | `langchain-core` LCEL only | Explicit runnables; the meta-package's agents are unused |
| LLM boundary | `LlmProvider` protocol | Unit-testable without a network; provider-swappable |
| API | FastAPI application factory | Endpoints stay 1–3 lines; logic lives in the pipeline |

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | — | Liveness and configured environment |
| `POST /v1/query` | — | Grounded answer with citations, or a structured abstention |
| `POST /v1/query/stream` | — | The same answer, preceded by SSE stage markers |
| `POST /v1/ingest` | `X-API-Key` | Authenticated ingestion request (acknowledgement; see below) |
| `POST /v1/feedback` | — | Record a reader's up/down verdict on an earlier answer |

Every response carries an `X-Request-ID`; every error is JSON containing that id. Interactive
docs with worked examples are at `/docs`.

`POST /v1/query/stream` emits `routed` → `retrieved` → `generating` progress markers and then
one `answer` event with the full payload. It is **not** token streaming: structured-output
parsing returns the object at once, and keeping the citation and abstention guarantees was
judged more valuable than a token-by-token effect.

`POST /v1/ingest` currently authenticates and records the request without running ingestion — a
60-second re-embed does not belong inside an HTTP request. The auth boundary is real and tested;
the worker behind it is future work.

## Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop with the WSL 2 backend on Windows, or Docker Engine with Compose elsewhere
- An OpenAI API key (only for answering; retrieval and engine health work without one)

## Local setup

```powershell
Copy-Item .env.example .env
```

Fill in `.env`: pick a `POSTGRES_PASSWORD` and keep it in sync with `TURBOFAN_DATABASE_URL`,
and set `TURBOFAN_OPENAI_API_KEY`. Then:

```powershell
uv sync --locked --group dev
docker compose up -d --wait postgres
uv run --no-sync alembic upgrade head
```

Load the corpus and the sensor data (the first run downloads the embedding model and takes
about a minute):

```powershell
uv run --no-sync python scripts/ingest_700_char_corpus.py
uv run --no-sync python scripts/ingest_fd001.py
```

Run the API:

```powershell
uv run --no-sync uvicorn turbofan_copilot.api.app:create_app --factory --app-dir src --reload
```

## Command-line tools

```powershell
uv run --no-sync python scripts/ask.py "How does a magnetic chip detector work?"
uv run --no-sync python scripts/summarise_engine.py test 5
uv run --no-sync python scripts/evaluate.py
```

`scripts/evaluate.py` is the single evaluation entry point. It scores retrieval offline, adds
abstention accuracy and token cost when a key is present, writes a timestamped report to
`data/evaluation/reports/`, and diffs it against the committed
[`data/evaluation/baseline.json`](data/evaluation/baseline.json), exiting non-zero on a quality
regression:

```
metric                            baseline   candidate       delta
retrieval.hit_at_1                  0.7778      0.5556     -0.2222  REGRESSED
retrieval.mean_reciprocal_rank      0.7778      0.7222     -0.0556  REGRESSED
abstention.accuracy                      1           1          +0
timing.retrieval_seconds            0.5060      0.0220     -0.4840
```

Use `--retriever lexical` to score a different configuration, and `--set-baseline` to promote
a run to the new reference point.

## Quality checks

```powershell
uv run --no-sync ruff check src tests migrations scripts
uv run --no-sync ruff format --check src tests migrations scripts
uv run --no-sync mypy src tests migrations scripts
uv run --no-sync pytest
```

Around 200 tests. Unit tests use fakes throughout — no model load, no database, no network.
Integration tests skip themselves when PostgreSQL or the API key is absent.

## Stop local services

```powershell
docker compose down
```

`docker compose down --volumes` deletes the database volume; that cannot be undone.

## Project documents

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — scope, the full dated decision log, evaluation
  plans, safety boundaries, and the milestone roadmap.
- [HANDOFF.md](HANDOFF.md) — compact current state and environment notes.

## Windows, OneDrive, and Smart App Control

This repository lives inside a OneDrive-synchronised folder, which caused two problems worth
recording because the fixes are not obvious.

**OneDrive re-materialises files, which re-triggers security evaluation.** Packages with native
extensions (`torch`, `tokenizers`, `regex`) would intermittently fail to import with *"An
Application Control policy has blocked this file"*, and uv could not hardlink from its cache
("The cloud operation cannot be performed on a file with incompatible hardlinks"). The fix is to
keep the environment out of the synchronised tree entirely. `.venv` here is a **directory
junction**, so every tool still finds it at the usual path while OneDrive skips the reparse
point:

```powershell
# one-time, from the project root
Remove-Item .venv -Recurse -Force
$target = "$HOME\.venvs\turbofan-maintenance-copilot"
uv sync --locked --group dev            # with $env:UV_PROJECT_ENVIRONMENT = $target
New-Item -ItemType Junction -Path .venv -Target $target
```

After that, `uv sync`, `uv run`, and `pytest` all work with no environment variables set.

**Smart App Control blocks brand-new unsigned DLLs.** It is `ENFORCED` on this machine
(`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1`) and
refuses to load native extensions that have no cloud reputation yet — which a package released
days ago will not have. `regex 2026.9.3` was blocked outright, breaking every `transformers`
import; `regex 2025.11.3` loads fine. `[tool.uv] constraint-dependencies = ["regex<2026"]` in
`pyproject.toml` holds it to an established build, and can be removed once the newer releases
have gained reputation.
