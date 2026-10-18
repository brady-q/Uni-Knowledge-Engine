# Uni Knowledge Engine

Retrieval-augmented question answering over university course PDFs, where every
claim in an answer is cited back to a specific document and page.

> **Status: M2 (schema and course CRUD) complete.** The service skeleton, the
> full content schema, ORM models, repositories and a working course CRUD API
> are in place and verified. The RAG pipeline — upload, extraction, chunking,
> embeddings, retrieval and answer generation — is not implemented yet. See
> [Roadmap](#roadmap).

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
codes, headers); services hold pipeline logic, own the transaction boundary and
are testable as plain Python; repositories own all SQL. Nothing below the
router layer imports FastAPI, which is why a service can be driven from a test,
a script or a background task. Embedding and LLM access sit behind narrow
`Protocol` interfaces so tests substitute deterministic fakes and never touch
the network.

The `ANTHROPIC_API_KEY` is only ever given to the `api` service. The `web`
service does not receive it, so it cannot reach the browser bundle.

## Quickstart

Requires Docker and Docker Compose. Nothing else.

```bash
cp .env.example .env     # or just run `make up`, which does this for you
make up                  # build and start db, api, web
make migrate             # apply migration 0001
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
make test                              # 73 tests, in-container, no network
make lint                              # ruff + mypy --strict
```

Expected readiness output once migration `0001` has been applied:

```json
{
  "status": "ok",
  "checks": {
    "db": { "status": "ok", "detail": null },
    "pgvector": { "status": "ok", "detail": "installed 0.8.6" }
  }
}
```

Before `make migrate`, `pgvector` instead reports
`"available 0.8.6, not yet installed"` — still `ok`, because the extension only
has to be *installable* until the first migration creates it.

Create a course:

```bash
curl -sX POST localhost:8000/api/v1/courses \
  -H 'content-type: application/json' \
  -d '{"code":"CS229","title":"Machine Learning"}' | jq
```

Run `make help` for every available target.

### Host-side development (optional)

The container is the source of truth, but editors want a local interpreter:

```bash
make venv          # backend/.venv from python3.12
make migrate-local # apply migrations from the host against localhost Postgres
make test-local    # runs the suite from the host against localhost Postgres
```

`make revision` runs in the container as the host user, so the migration it
generates is owned by you and not by the image's non-root uid.

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

## Data model

Migration `0001` creates the whole content schema. Three tables, related by
foreign key rather than by nesting:

```
courses                  documents                      chunks
─────────                ──────────                     ──────
id          uuid pk      id          uuid pk            id          uuid pk
code        unique  ◀────course_id   fk cascade    ◀────document_id fk cascade
title                    filename                       ordinal
description              content_hash                   page_start
created_at               byte_size                      page_end
updated_at               page_count  null               text
                         status      checked            embedding   vector(384) null
                         error       null
                         created_at
                         updated_at
```

The choices worth knowing about:

- **`courses.code` is UNIQUE, but the primary key is a UUID.** Course codes get
  renamed and reused; identity must never change. Codes are normalised to upper
  case at the API boundary, so `cs229` and `CS229` collide and no
  case-insensitive column type is needed.
- **Both foreign keys are `ON DELETE CASCADE`, declared in the database.** A
  chunk whose document is gone is a citation that can never be resolved.
  Deleting a course is one `DELETE`, not an application-level cleanup someone
  has to remember. The ORM relationships use `passive_deletes=True` so they
  trust the database rather than loading every child to delete it individually.
- **Foreign keys are not automatically indexed by Postgres**, so each one is
  covered deliberately: `uq_documents(course_id, content_hash)` and
  `uq_chunks(document_id, ordinal)` both lead with the foreign key column, which
  serves lookups by parent as well as enforcing uniqueness.
- **`chunks.embedding` is nullable.** Chunks are written during extraction and
  embedded in a later pass, so a chunk without a vector is a valid intermediate
  state rather than corruption.
- **`documents.status` is a checked `VARCHAR`, not a native `ENUM`.** Adding a
  pipeline stage later is then a CHECK constraint swap inside a normal
  transaction, instead of an `ALTER TYPE` that cannot always run in one.
- **Page numbers are constrained in the database** (`page_start >= 1`,
  `page_end >= page_start`). These are the values the citation guarantee rests
  on, so they are enforced rather than assumed from the chunker.
- **The HNSW index is built while the table is empty**, which is free; building
  it after ingestion would mean a long lock on a large table. Its opclass is
  `vector_cosine_ops` to match the `<=>` operator retrieval will use — an index
  built for the wrong distance function is silently never used.

## Course API

| Method   | Path                    | Success | Failures                    |
| -------- | ----------------------- | ------- | --------------------------- |
| `POST`   | `/api/v1/courses`       | `201` + `Location` | `409` duplicate code, `422` |
| `GET`    | `/api/v1/courses`       | `200`   | `422` bad paging            |
| `GET`    | `/api/v1/courses/{id}`  | `200`   | `404`, `422` bad id         |
| `PATCH`  | `/api/v1/courses/{id}`  | `200`   | `404`, `409`, `422`         |
| `DELETE` | `/api/v1/courses/{id}`  | `204`   | `404`                       |

`PATCH` is a true partial update: an absent key leaves the column alone, while
an explicit `null` clears a nullable one. `null` for `code` or `title` is a
`422`, because those columns are `NOT NULL` — the DTO refuses it rather than
letting the database do it.

List responses are paginated and carry the full count, so a client can render
"51–100 of 240" without a second request:

```json
{ "items": [ ... ], "total": 240, "limit": 50, "offset": 50 }
```

Results are ordered by `code`. An unordered `LIMIT`/`OFFSET` has no defined row
order, so paging through it can repeat or skip rows.

### Uniqueness is enforced by the database, not checked first

There is deliberately no "does this code already exist?" query before the
insert. Two concurrent requests can both pass that check before either commits.
The `UNIQUE` index is evaluated atomically, so the only correct order is to
attempt the write and translate the rejection:

```
IntegrityError → constraint name is uq_courses_code → ConflictError → 409
```

The constraint name is read from the driver error rather than matched against
the message text, so a table with two unique constraints can never report one
violation as the other. `tests/integration/test_courses_crud.py` fires five
identical creates concurrently and asserts on exactly one `201` and four `409`s.

Domain errors carry no HTTP knowledge: services raise `NotFoundError` or
`ConflictError` from `app/errors.py`, and handlers registered in `create_app`
map them to statuses. That is why no route handler contains a `try`/`except`.

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
│   ├── migrations/
│   │   ├── env.py           # async; target_metadata + pgvector rendering
│   │   └── versions/0001_initial_schema.py
│   ├── app/
│   │   ├── main.py          # create_app factory, lifespan, error handlers
│   │   ├── config.py        # pydantic-settings
│   │   ├── db.py            # async engine, session dependency
│   │   ├── models.py        # ORM models = the source of truth for all DDL
│   │   ├── schemas.py       # Pydantic DTOs (separate from the models)
│   │   ├── errors.py        # domain errors; no HTTP, no FastAPI
│   │   ├── routers/         # health, courses
│   │   ├── services/        # health, courses
│   │   └── repositories/    # every SQL statement in the application
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
make test        # in-container: unit + integration, 73 tests
make test-local  # host venv; integration tests skip if no local Postgres
make check       # lint + test + frontend typecheck (what CI would run)
```

51 unit tests, 22 integration. Deliberate choices:

- **Unit tests point at `127.0.0.1:1`**, a port nothing can be listening on.
  This makes it impossible for a unit test to quietly depend on a live
  database, and it means the readiness *failure* path is covered end-to-end
  with no infrastructure at all.
- **`httpx.ASGITransport` with a manual `lifespan_context`** drives the app
  in-process — no live server to start, but startup logic (engine creation,
  storage directory) is still exercised.
- **Integration tests run against a dedicated `uke_test` database**, created
  and migrated on demand next to the development one. They truncate tables
  between cases, and doing that to the database someone is developing against
  would be hostile. Migrations are applied by shelling out to the real
  `alembic upgrade head`, so the suite proves the *migration* works and not
  merely that the models agree with themselves.
- **`alembic check` is a test.** `test_schema.py` fails if the models, the
  migration and the live schema disagree, which is what keeps `make revision`
  trustworthy for the next change instead of a source of surprise diffs. The
  other schema tests query the Postgres catalogue or attempt an illegal write,
  rather than trusting the ORM's own picture of the world.
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

**Alembic owns all DDL.** `Base.metadata.create_all` is never used, in tests or
anywhere else — the test database is built by running the real migration. The
URL comes from application settings rather than `alembic.ini`, so the app and
its migrations cannot disagree about which database they mean, and no
credentials sit in a committed file.

**Autogenerate is a draft, not an answer.** It reliably detects added tables
and columns, and reliably misses `CREATE EXTENSION`, data migrations and
renames — it sees a rename as a drop plus an add, which silently destroys data.
Migration `0001` was generated and then hand-edited to put
`CREATE EXTENSION IF NOT EXISTS vector` before the table that uses the type.
`make revision` prints this reminder.

**Constraints are named by convention, not by Postgres.** Left to itself
Postgres invents names like `courses_code_key`, and those names are what a
later `ALTER` or a `downgrade` has to reference. A `MetaData`
naming convention means every constraint name is derivable from `models.py` —
which is also what lets the 409 mapping match on `uq_courses_code` exactly.

**UUID primary keys, not `bigserial`.** These ids appear in frontend routes and
in citation payloads, where a sequential integer leaks how many rows exist and
invites enumeration. The cost is 16 bytes and unordered index inserts, which is
irrelevant at this scale. They are generated in Python *and* defaulted
server-side, so data migrations and hand-written SQL do not have to invent one.

**Invariants live in the schema; services handle the violation.** A `UNIQUE`
index is atomic and unconditional; a check-then-insert in Python is a race two
concurrent requests can both win. See
[Course API](#uniqueness-is-enforced-by-the-database-not-checked-first).

**`eager_defaults=True` on the mapper.** Server-generated timestamps are
fetched back with `RETURNING` rather than left expired, so a router can
serialise a just-committed row without a second `SELECT` — which, paired with
`expire_on_commit=False`, is what keeps a write to one round trip.

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
| **M2**    | Content schema in migration `0001`, models, repositories, course CRUD — **done** |
| M3        | PDF upload, PyMuPDF extraction with page metadata, chunking     |
| M4        | Sentence-Transformers embeddings, HNSW retrieval, `/search`     |
| M5        | Cited answers via Claude, marker resolution, query log table    |
| M6        | Course and Q&A screens, PDF pane that jumps to the cited page   |
| M7        | Golden-set eval harness (recall@k, citation precision), polish  |
