# Turbofan Maintenance Intelligence Copilot

A retrieval-augmented maintenance assistant for turbine engines. It answers questions from
the FAA *Aviation Maintenance Technician Handbook: Powerplant* with **page-level citations**,
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
- **Engine numbers never come from the LLM.** "Engine 5 shows rising EGT during start. What
  should I inspect?" attaches a least-squares sensor-trend summary and a remaining-useful-life
  prediction from a fitted gradient-boosted model. The prediction carries its typical error and
  is labelled as a prediction, clearly separated from the manual's guidance.

## Measured results

Retrieval over nine frozen, page-grounded questions (top-5 budget):

| Config | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Lexical (IDF baseline) | 0.556 | **0.889** | **0.889** | 0.722 |
| BGE semantic | 0.667 | 0.889 | 0.889 | 0.759 |
| **RRF hybrid (shipped)** | **0.778** | 0.778 | 0.778 | **0.778** |

The hybrid wins on Hit@1 and MRR and is the shipped configuration. It *loses* Hit@3/5 to the
lexical baseline: two paraphrased questions fall to ranks 7 and 11. That is reported here
rather than hidden.

- **Abstention:** 10/10 on 10 labelled cases (4 answerable, 3 out-of-corpus, 3 injection).
- **Cost:** a full evaluation run is roughly US$0.002 on `gpt-4o-mini`.

Remaining useful life on the 100 held-out FD001 test engines, with the model chosen on a
separate validation split grouped by engine:

| Model | MAE | RMSE | Mean error |
|---|---|---|---|
| Naive constant | 34.8 | 42.0 | +12.7 |
| Degradation-index baseline | 25.8 | 32.7 | +25.8 |
| Ridge on windowed sensor features | 12.6 | 15.4 | +1.5 |
| **Gradient-boosted trees (shipped)** | **8.4** | **11.5** | **+1.8** |

These scores cap the true remaining life at 125 cycles, the usual C-MAPSS convention. Against
the uncapped targets, the way published FD001 results are usually reported, the shipped model
scores **MAE 9.5 / RMSE 12.8**. The old baseline never once predicted less remaining life than
an engine really had (its mean error equals its MAE). Overestimating remaining life is the unsafe
direction for maintenance, because it schedules the inspection too late. The shipped model errs
both ways (38 engines under, 60 over, 2 exact) with a mean error under 2 cycles. Three checks
back the result: training on shuffled labels collapses it to the naive score, removing engine
age as a feature barely changes it, and the evaluation harness reproduces the old baseline's
published numbers exactly.

**Honest limits.** Nine retrieval cases and ten abstention cases are a smoke test, not a
benchmark: one case is worth 11% of a retrieval metric. The abstention cases are textbook
injections, not adversarial ones. The RUL model is trained and tested only on FD001, the
simplest C-MAPSS subset (one operating condition, one fault mode), and the error it reports
with each prediction is an average over the test set, not a calibrated interval for that
engine.

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img src="docs/architecture-light.svg" width="100%" alt="Architecture. A question goes from the browser into a FastAPI container, which retrieves handbook passages (lexical plus BGE, fused with RRF), adds engine data when an engine is named, sends a prompt to the OpenAI API for a structured reply, and maps the reply's passage numbers to handbook pages. There are two refusal exits: nothing retrieved, and the model declines. PostgreSQL with pgvector holds the chunks and vectors (loaded at startup), the sensor readings, and a log of every query. Offline scripts fill it from the FAA handbook and NASA C-MAPSS FD001.">
</picture>

One question's path. Every step inside the container is deterministic code except the highlighted
call to the OpenAI API, which writes the prose and picks passage numbers; the citations and the
engine figures come from code and data. Dashed exits are the two refusal points. The diagram is
generated by `docs/build_architecture.py` in matching light and dark versions.

| Layer | Choice | Why |
|---|---|---|
| Chunking | 700 chars / 100 overlap, page-contained | Page provenance is what makes a citation checkable |
| Embeddings | `BAAI/bge-small-en-v1.5`, pinned revision, ONNX Runtime on CPU | Reproducible, no embedding API cost, no PyTorch |
| Vector store | PostgreSQL + pgvector, HNSW `vector_cosine_ops` | Relational and vector data in one place |
| Fusion | Reciprocal rank fusion, `k = 60` | No score normalisation needed across two scales |
| Orchestration | `langchain-core` LCEL only | Explicit runnables; the meta-package's agents are unused |
| LLM boundary | `LlmProvider` protocol | Unit-testable without a network; provider-swappable |
| RUL model | scikit-learn `HistGradientBoostingRegressor` on 30-cycle windowed sensor features | Cut the baseline's MAE by two thirds and removed its over-prediction bias |
| API | FastAPI application factory | Endpoints stay 1 to 3 lines; logic lives in the pipeline |

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /` | None | Browser chat page (see below) |
| `GET /health` | None | Liveness: is the process up |
| `GET /ready` | None | Readiness: can this instance serve queries (see below) |
| `POST /v1/query` | None | Grounded answer with citations, or a structured abstention |
| `POST /v1/query/stream` | None | The same answer, preceded by SSE stage markers |
| `POST /v1/feedback` | None | Record a reader's up/down verdict on an earlier answer |

Every response carries an `X-Request-ID`; every error is JSON containing that id. Interactive
docs with worked examples are at `/docs`.

The query endpoints are open because this runs locally. Each query spends the configured OpenAI
key, so they need their own API key before any public deployment.

`POST /v1/query/stream` emits `routed` → `retrieved` → `generating` progress markers and then
one `answer` event with the full payload. It is **not** token streaming: structured-output
parsing returns the object at once, and keeping the citation and abstention guarantees was
judged more valuable than a token-by-token effect.

There is no ingestion endpoint. The corpus and the sensor data are loaded once, offline, by
`scripts/ingest_700_char_corpus.py` and `scripts/ingest_fd001.py`; a deployed instance only
reads them, which keeps an unused write path off the public surface.

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

Over 240 tests. Unit tests use fakes throughout: no model load, no database, no network.
Integration tests skip themselves when PostgreSQL or the API key is absent.

## Chat in the browser

With the stack running, open **http://localhost:8000/** for a chat page. It streams
the pipeline's stages as they happen (routing, retrieving, generating), then shows
the answer with its FAA handbook sources. Engine questions add a card with the
remaining-useful-life prediction, its typical error, and the sensors moving most. A
question the evidence cannot answer shows as a refusal rather than a guess.

The page is one self-contained HTML file served by the API itself: no build step, no
external assets, no extra dependency, and no second server. It reads the
`POST /v1/query/stream` response by hand, because the browser's `EventSource` only
supports `GET`. Everything the model returns is inserted as text, never as HTML,
because answers quote retrieved documents and those are treated as untrusted.

## Liveness and readiness

Two endpoints, answering two different questions.

- `GET /health`: **liveness**. Is the process up? It touches nothing else, so it
  keeps answering 200 during a database outage. A failure here means *restart this
  container*. The image's `HEALTHCHECK` uses this one.
- `GET /ready`: **readiness**. Can this instance serve queries? It checks the
  database, that the corpus has been ingested, that the embedding weights are on
  disk, and that a generation key is configured, and answers 503 if any of them
  fails. A failure means *stop sending traffic here*, not *restart me*.

```json
{"status": "ready", "checks": [
  {"name": "database",        "ok": true, "detail": "reachable"},
  {"name": "corpus",          "ok": true, "detail": "1050 chunks"},
  {"name": "embedding_model", "ok": true, "detail": "present"},
  {"name": "llm_credentials", "ok": true, "detail": "configured"}]}
```

Every check is reported whether it passed or not, so one call explains a bad
deployment. Details stay terse (an exception type, a row count) because the
endpoint is unauthenticated and a driver's error text can contain the connection
string.

Keeping them separate matters: probing the database from a liveness check would
turn a brief outage into a restart loop across every replica.

## Running the API in a container

The whole stack comes up with one command. The API image is built from
`Dockerfile`; PostgreSQL starts first and the API waits for its health check.

```powershell
docker compose up -d --build
curl http://localhost:8000/health
```

The database still needs its schema and data the first time:

```powershell
docker compose exec api alembic upgrade head
```

Then ingest the corpus and FD001 from the host (`scripts/ingest_700_char_corpus.py`
and `scripts/ingest_fd001.py`), which need the raw PDFs and data files that are
deliberately not in the image.

Notes on the image:

- **Two stages.** The builder resolves the locked dependencies and downloads the
  pinned embedding weights; the runtime keeps only the finished virtual
  environment, those weights, and the migrations. No uv, no compilers, no source
  tree, and no git history ship.
- **Non-root.** The service runs as uid 1001 with no login shell.
- **No PyTorch.** Every question still has to be embedded by the same model that
  embedded the passages, so the model cannot be dropped, but PyTorch was only the
  program running it. The embedder uses ONNX Runtime and the ONNX export that BGE's
  authors publish for the pinned revision. Before PyTorch was removed, all 1,050
  stored passage vectors were re-embedded and compared: the worst cosine similarity
  was 0.9999998 and no single value moved by more than 0.00000024. The evaluation
  then reproduced the baseline exactly, down to the same 10,080 prompt tokens.
- **The weights are baked in**, so a container starts without reaching Hugging
  Face (`HF_HUB_OFFLINE=1`) and the first request does not pay a download.
- **Migrations are not applied at startup.** Several replicas booting together
  must not race to migrate one database, so `alembic upgrade head` is a separate,
  deliberate step.
- **Size:** a 328 MB compressed pull and 947 MB on disk, with about 490 MiB of memory
  in use once the pipeline is loaded. Replacing PyTorch roughly halved the image.
  (`docker images` reports a larger figure because it also counts the
  multi-platform manifest and build attestations.)

## Stop local services

```powershell
docker compose down
```

`docker compose down --volumes` deletes the database volume; that cannot be undone.

## Environment note

Some packages ship compiled extensions that Windows Smart App Control will refuse to load until
a release has built up reputation. `pyproject.toml` therefore pins two of them
(`[tool.uv] constraint-dependencies = ["regex<2026", "scipy<1.18"]`); without those, `transformers`
and `scikit-learn` can fail to import on Windows. Both constraints can be dropped once the newer
releases are widely established.

Keep the repository and its virtual environment outside a cloud-synchronised folder. A sync engine
that dehydrates and re-materialises files makes native extension DLLs look new on every access,
which triggers the problem above and also breaks uv's hardlinking from its cache.
