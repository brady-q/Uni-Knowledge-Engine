# Uni Knowledge Engine

Retrieval-augmented question answering over university course PDFs, where every
claim in an answer is cited back to a specific document and page.

> **Status: M1 (scaffolding) complete.** The service skeleton, database, Docker
> environment and health checks are in place and verified. The RAG pipeline —
> upload, extraction, chunking, embeddings, retrieval and answer generation —
> is not implemented yet. See [Roadmap](#roadmap).

---

## The core design idea

The property this project is built to guarantee is that **citations cannot be
hallucinated**. The language model never writes a page number.

Retrieved chunks are injected into the prompt labelled `[S1]…[S6]`. The model
returns structured JSON citing only those markers. The backend then resolves
each marker back to the chunk row it came from and reads the document title and
page range **out of the database**. A page number that was not in the retrieved
set is structurally impossible to emit, and any marker the model invents is
dropped and logged.

Everything else in the architecture — the typed boundaries, the query log, the
planned eval harness — exists to make that claim testable rather than asserted.

## Architecture

```
┌────────────────┐        ┌──────────────────────────────┐        ┌──────────────┐
│  Next.js web   │  HTTP  │        FastAPI api           │  SQL   │  Postgres 17 │
│  (App Router)  │───────▶│  routers → services → repos  │───────▶│  + pgvector  │
└────────────────┘        └──────────────────────────────┘        └──────────────┘
                                        │
                                        │ Claude API (server-side only)
                                        ▼
```

Dependencies inside the API point one way only: **routers → services →
repositories → models**. Routers handle HTTP concerns (validation, status
codes); services hold pipeline logic and are testable as plain Python;
repositories own all SQL. Embedding and LLM access sit behind narrow `Protocol`
interfaces so tests substitute deterministic fakes and never touch the network.

The `ANTHROPIC_API_KEY` is only ever given to the `api` service. The `web`
service does not receive it, so it cannot reach the browser bundle.

## Quickstart

Requires Docker and Docker Compose. Nothing else.

```bash
cp .env.example .env     # or just run `make up`, which does this for you
make up                  # build and start db, api, web
```

| Service | URL                                  |
| ------- | ------------------------------------ |
| Web     | http://localhost:3000                |
| API     | http://localhost:8000                |
| Docs    | http://localhost:8000/docs           |

Verify the stack:

```bash
curl -s localhost:8000/health          | jq
curl -s localhost:8000/health/ready    | jq
make test                              # 20 tests, in-container, no network
make lint                              # ruff + mypy --strict
```

Expected readiness output before any migrations exist:

```json
{
  "status": "ok",
  "checks": {
    "db": { "status": "ok", "detail": null },
    "pgvector": { "status": "ok", "detail": "available 0.8.6, not yet installed" }
  }
}
```

Run `make help` for every available target.

### Host-side development (optional)

The container is the source of truth, but editors want a local interpreter:

```bash
make venv          # backend/.venv from python3.12
make test-local    # runs the suite from the host against localhost Postgres
```

## Health endpoints

Two endpoints, because they answer different questions:

| Endpoint        | Question                          | Touches the database |
| --------------- | --------------------------------- | -------------------- |
| `/health`       | Is this process alive?            | **No**               |
| `/health/ready` | Should traffic be routed here?    | Yes                  |

`/health` deliberately checks nothing external. If liveness depended on
Postgres, a transient database outage would cause an orchestrator to kill a
perfectly healthy API container and make the incident worse.

`/health/ready` runs one probe per dependency and returns **503** if any fails,
with a per-component reason:

```json
{
  "status": "error",
  "checks": {
    "db": { "status": "error", "detail": "OperationalError: connection refused" },
    "pgvector": { "status": "ok", "detail": "installed 0.8.6" }
  }
}
```

The probes convert failures into values instead of raising, so an unhealthy
dependency produces an informative 503 rather than an opaque 500. The
`pgvector` probe exists to catch running against a plain `postgres` image at
startup instead of when the first migration fails.

Both endpoints are unversioned: probes should not have to track the API
version. Feature routers mount under `/api/v1`.

## Project layout

```
├── docker-compose.yml       # db (pgvector/pg17), api, web
├── Makefile                 # every workflow: up, test, lint, migrate, ...
├── .env.example             # the only committed env file
├── backend/
│   ├── Dockerfile           # python:3.12-slim, non-root
│   ├── requirements.txt     # runtime deps, exactly pinned
│   ├── requirements-dev.txt # test/lint deps
│   ├── pyproject.toml       # ruff + mypy + pytest config only
│   ├── alembic.ini
│   ├── migrations/          # async env.py; zero revisions until M2
│   ├── app/
│   │   ├── main.py          # create_app factory + lifespan
│   │   ├── config.py        # pydantic-settings
│   │   ├── db.py            # async engine, session dependency
│   │   ├── schemas.py       # Pydantic DTOs
│   │   ├── routers/health.py
│   │   └── services/health.py
│   └── tests/
│       ├── unit/            # no database, no network
│       └── integration/     # real Postgres; auto-skips when unreachable
└── frontend/
    ├── Dockerfile           # next dev against the bind mount
    └── src/
        ├── app/             # App Router: layout, status page
        └── lib/             # typed API client + DTO mirrors
```

## Testing

```bash
make test        # in-container: unit + integration, 20 tests
make test-local  # host venv; integration tests skip if no local Postgres
make check       # lint + test + frontend typecheck (what CI would run)
```

Three deliberate choices:

- **Unit tests point at `127.0.0.1:1`**, a port nothing can be listening on.
  This makes it impossible for a unit test to quietly depend on a live
  database, and it means the readiness *failure* path is covered end-to-end
  with no infrastructure at all.
- **`httpx.ASGITransport` with a manual `lifespan_context`** drives the app
  in-process — no live server to start, but startup logic (engine creation,
  storage directory) is still exercised.
- **Markers `integration`, `slow`, `live`** separate tests by what they cost.
  `slow` (real embedding model) and `live` (real Claude API) tests will exist
  but stay out of the default run: a suite that downloads models and spends
  money is a suite people stop running.

## Decisions and tradeoffs

**Python 3.12, not the host's 3.14.** `torch`/`sentence-transformers` cp314
wheels are unreliable and fall back to multi-hour source builds. Committing to
3.12 now avoids a painful re-pin when embeddings land.

**`requirements*.txt` with `==` pins, not `pyproject` dependencies.** Neither
`uv` nor `poetry` is available here, and pip has no lockfile; exact pins are the
closest substitute. Splitting runtime from dev deps also gives real Docker
layer caching. `pyproject.toml` holds tool configuration only, because this is
an application, not a distributable package.

**No module-level `app` object.** The API is served with
`uvicorn app.main:create_app --factory`. A module-level `app = create_app()`
would mean that merely *importing* `app.main` reads the environment, so tests
and Alembic would need a configured `DATABASE_URL` just to import it.

**Startup does not require the database.** The lifespan creates the engine but
never connects. An API that boots and honestly reports its own unreadiness is
more useful than one that crash-loops while Postgres is still starting.

**Alembic owns all DDL, from the first revision.** `Base.metadata.create_all`
is never used outside tests. Migrations are wired up now, with zero revisions,
so the first schema change is a normal `make revision` rather than a
retrofit. The URL comes from application settings rather than `alembic.ini`, so
the app and its migrations cannot disagree about which database they mean — and
no credentials sit in a committed file.

**Readiness reports are a FastAPI dependency.** `get_readiness_report` is
overridable, which is what lets the endpoint's contract be tested without a
database and lets an `embedder` check be added in exactly one place later.

**No UI framework yet.** The status page is one CSS file. Choosing a design
system before there are screens is a bet with no payoff; the real UI arrives
with the course and Q&A views.

**Two frontend base URLs.** Server Components resolve the API by its compose
service name (`http://api:8000`); the browser can only reach the published port
(`http://localhost:8000`). Collapsing these into one variable breaks under
Docker.

**Known limitation, deferred deliberately:** ingestion will run on FastAPI
`BackgroundTasks` rather than a task queue, so in-flight work will not survive
an API restart. Swapping in an `arq` worker is a later milestone. A smaller
working system beats a half-finished Celery setup.

## Roadmap

| Milestone | Scope                                                          |
| --------- | -------------------------------------------------------------- |
| **M1**    | Scaffolding, Docker, health checks — **done**                   |
| M2        | Full schema in migration `0001`, models, repositories, course CRUD |
| M3        | PDF upload, PyMuPDF extraction with page metadata, chunking     |
| M4        | Sentence-Transformers embeddings, HNSW retrieval, `/search`     |
| M5        | Cited answers via Claude, marker resolution, query logging      |
| M6        | Course and Q&A screens, PDF pane that jumps to the cited page   |
| M7        | Golden-set eval harness (recall@k, citation precision), polish  |
